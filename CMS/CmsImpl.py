import numpy as np
from numba import jit
import sys


# 调用函数计算平方范数
def l2_squared(left, right=None, cuda=True):
    if cuda:
        from ._CMS_ImplCuda import cuda_l2_squared
        return cuda_l2_squared(left, right)
    else:
        from ._CMS_ImplNumba import numba_l2_squared
        return numba_l2_squared(left, right)


# 在约束条件下更新权重矩阵（高斯核）
@jit(nopython=True)
def _rbf_reduce_mul_impl(mode_mode_dist_sq, constraint_list, h_global, result, truncate, scale):
    for x, y in constraint_list:
        # 计算约束拟合度，并根据比例缩放
        constraint_fit = np.sqrt(mode_mode_dist_sq[x, y]) * scale
        # 防止约束拟合度过小，最小值设置为0.001
        constraint_fit = max(constraint_fit, 0.001)
        # 约束拟合度不能超过全局带宽h_global
        constraint_fit = min(h_global, constraint_fit)

        # 计算x和y的核密度估计值
        x_ks = np.exp(-mode_mode_dist_sq[x] / (constraint_fit ** 2))
        y_ks = np.exp(-mode_mode_dist_sq[y] / (constraint_fit ** 2))
        # 将小于截断值的核密度估计值设为0（去除小的核密度估计）
        x_ks[x_ks < truncate] = 0
        y_ks[y_ks < truncate] = 0

        # 找到非零的y核密度估计值的索引
        buf = np.nonzero(y_ks)[0]

        # 更新结果矩阵
        for a in range(len(mode_mode_dist_sq)):
            if x_ks[a] == 0:
                continue
            for b in buf:
                result[a, b] = result[a, b] * (1 - x_ks[a] * y_ks[b])


# 在约束条件下更新权重矩阵（球核）
@jit(nopython=True)
def _ball_reduce_mul_impl(mode_mode_dist_sq, constraint_list, h_global, result, scale):
    for x, y in constraint_list:
        constraint_fit = np.sqrt(mode_mode_dist_sq[x, y]) * scale
        # Protect against underrunning a sensible fit if constraints are very close, 0 leads to arithmetic error later
        constraint_fit = max(constraint_fit, 0.001)
        constraint_fit = min(h_global, constraint_fit)

        x_ks = mode_mode_dist_sq[x] <= constraint_fit ** 2
        y_ks = mode_mode_dist_sq[y] <= constraint_fit ** 2

        buf = np.nonzero(y_ks)[0]

        for a in range(len(mode_mode_dist_sq)):
            if x_ks[a] == 0:
                continue
            for b in buf:
                result[a, b] = result[a, b] * (1 - x_ks[a] * y_ks[b])


class CMS:
    def __init__(self, h, max_iterations=1000, blurring=True, kernel=.02, use_cuda=False, c_scale=.5, label_merge_k=.95,
                 label_merge_b=.1, stop_early=True, verbose=True, save_history=True):
        """
        约束均值漂移聚类算法

        :param h: 带宽参数，可以是一个标量或一个函数，函数根据迭代次数返回带宽
        :param max_iterations: 最大迭代次数
        :param blurring: 是否使用模糊均值漂移
        :param kernel: 如果是“ball”，则使用ball内核。如果标量在[0, 1)范围内浮动，则使用截断高斯核，以kernel的值作为截断边界
        :param use_cuda: 是否使用CUDA加速
        :param c_scale: 约束缩放参数
        :param label_merge_k: 用于标签提取的连接矩阵的参数，表示两种模式被标记为连接的最小接近度
        :param label_merge_b: 用于标签提取的连接矩阵的参数，表示两种模式永远不会被视为连接的最差约束减少。
        :param stop_early: 是否在聚类中心变得稳定时提前停止
        :param verbose: 是否输出详细信息
        :param save_history: 是否保存模式、带宽、核和约束权重的历史记录
        """
        if kernel != 'ball' and not (0 < kernel < 1):
            raise ValueError("Invalid kernel: {}".format(kernel))

        if c_scale <= 0:
            raise ValueError("Invalid constraint scale: {}".format(c_scale))

        self.h = h
        self.max_iterations = max_iterations
        self.blurring = blurring
        self.kernel = kernel
        self.use_cuda = use_cuda
        self.c_scale = c_scale
        self.label_merge_k = label_merge_k
        self.label_merge_b = label_merge_b
        self.stop_early = stop_early
        self.verbose = verbose
        self.save_history = save_history

        self.mode_history_ = None
        self.block_history_ = None
        self.kernel_history_ = None
        self.bandwidth_history_ = None
        self.labels_ = None
        self.modes_ = None

    def calculate_labels(self, modes, curr_h, inter_weights):
        """
        计算给定模式、带宽和约束权重的标签

        :param modes: Modes/cluster centers
        :param curr_h: Bandwidth used
        :param inter_weights: Reduction weights for the given modes and bandwidth
        :return: Label assignments
        """
        from scipy.sparse.csgraph import connected_components

        # 计算模式之间的距离
        mode_mode_dist_sqs = l2_squared(modes, cuda=self.use_cuda)

        # 根据核函数类型计算核密度权重
        if self.kernel == 'ball':
            mode_mode_kernel_weights = (mode_mode_dist_sqs <= curr_h ** 2).astype(np.float32)
        else:
            mode_mode_kernel_weights = np.exp(-mode_mode_dist_sqs / (curr_h ** 2))
            mode_mode_kernel_weights[mode_mode_kernel_weights < self.kernel] = 0

        # 确定允许合并的模式对
        assert mode_mode_kernel_weights.shape == inter_weights.shape
        allow_merge = np.logical_and(mode_mode_kernel_weights > self.label_merge_k, inter_weights > self.label_merge_b)

        # 使用连通组件算法计算标签
        return connected_components(allow_merge)[1]

    # 拟合数据并返回聚类中心（模式）
    def fit_transform(self, points, constraints):
        self.fit(points, constraints)
        return self.modes_

    # 拟合数据并返回标签
    def fit_predict(self, points, constraints):
        self.fit(points, constraints)
        return self.labels_

    # CMS的核心部分，包括初始化、迭代更新模式（聚类中心）和计算最终标签
    def fit(self, points, constraints):
        """
        Fit on the given data/sampling points and cannot-link constraints
        :param points: 一个NxD样本点数组，既作为初始采样点又作为初始聚类中心
        :param constraints: 一个Cx2数组索引不应链接的采样点
        """
        from .Constraints import constraint_list_from_constraints

        # 将约束转换为内部使用的格式
        constraint_list = constraint_list_from_constraints(constraints)
        # 将输入数据点转换为float32类型
        points = points.astype(np.float32)
        # 初始化模式为数据点
        modes = points

        # 如果需要保存历史记录，初始化相应的列表
        if self.save_history:
            self.mode_history_ = [modes]
            self.block_history_ = []
            self.kernel_history_ = []
            self.bandwidth_history_ = []

        # 检查带宽参数是否为固定值
        is_fixed_h = not callable(self.h)
        # 根据迭代次数计算当前带宽
        for epoch in range(self.max_iterations):
            curr_h = np.float32(self.h if is_fixed_h else self.h(epoch))
            if self.verbose:
                print('CMS: Iteration: {}, Bandwidth: {}'.format(epoch, curr_h), file=sys.stderr)
            assert not np.isnan(curr_h)
            if curr_h <= 0:
                raise ValueError("Require curr_h > 0, was {}".format(curr_h))

            # 计算模式之间的平方距离矩阵（根据是否为模糊CMS选择采样点）
            mode_mode_dist_sqs = l2_squared(modes, cuda=self.use_cuda)
            if self.blurring:
                # 从上一个模式采样
                sample_dist_sqs = mode_mode_dist_sqs
                sample_points = modes
            else:
                # 从原始数据点采样
                sample_dist_sqs = l2_squared(modes, points, cuda=self.use_cuda)
                sample_points = points

            # 根据核函数类型计算核权重
            if self.kernel != 'ball':
                kernel_weights = np.exp(-sample_dist_sqs / (curr_h ** 2))
                kernel_weights[kernel_weights < self.kernel] = 0
            else:
                kernel_weights = (sample_dist_sqs <= curr_h ** 2).astype(np.float32)

            # 初始化约束权重为全1
            # 根据核函数类型调用相应的约束权重计算函数
            inter_weights = np.ones((len(modes),) * 2, dtype=np.float32)
            assert np.all(np.logical_and(constraint_list >= 0, constraint_list < len(modes)))
            if self.kernel == 'ball':
                _ball_reduce_mul_impl(mode_mode_dist_sqs, constraint_list, curr_h, inter_weights, self.c_scale)
            else:
                _rbf_reduce_mul_impl(mode_mode_dist_sqs, constraint_list, curr_h, inter_weights, self.kernel,self.c_scale)
            if self.blurring:
                # Only enforce inter-weight diagonal 1 for cluster mode
                np.fill_diagonal(inter_weights, 1)
                # Only check symetry in cluster sample mode, in other mode cluster centers may drift unsymetrically
                assert not np.isnan(inter_weights).any()

            assert np.logical_and(inter_weights >= 0, inter_weights <= 1).all()
            # 计算有效权重
            eff_weights = kernel_weights * inter_weights
            # 计算权重总和
            weights_sum = np.sum(eff_weights, axis=-1, keepdims=True)
            # 如果某点完全失去吸引力，则重置
            weightless = (weights_sum <= 1e-10).nonzero()[0]
            if len(weightless) > 0:
                print("CMS: Warn: Iteration {}: Had {} cluster centers without weight".format(epoch, len(weightless)),
                      file=sys.stderr)
            # 更新模式
            modes_new = np.dot(eff_weights, sample_points) / weights_sum
            modes_new[weightless] = modes[weightless]

            # 检查模式是否收敛：如果满足提前停止条件，设置break_next标志
            break_next = False
            if self.stop_early:
                if np.allclose(modes_new, modes, rtol=1.e-4, atol=1.e-6):
                    if np.all(np.logical_or(inter_weights < self.label_merge_b, kernel_weights > self.label_merge_k)):
                        # Don't immediately break here, but add to history and then break. This is because we have already
                        # invested the time into the calculation anyways so we might actually return the first "over-settled"
                        # modes instead of the ones before. Because modes may have still slightly changed on the iteration
                        # where we detect settling, this can actually improve cluster assignment on some edge cases
                        break_next = True
                    elif is_fixed_h:
                        # If h is fixed, can stop immediately when no movement occurs, because next iteration will have
                        # same result
                        break_next = True

            # 保存历史记录
            if self.save_history:
                self.mode_history_.append(modes_new)
                self.block_history_.append(inter_weights)
                self.kernel_history_.append(kernel_weights)
                self.bandwidth_history_.append(curr_h)

            # 更新模式
            modes = modes_new

            # 提前退出循环
            if break_next:
                break

        # 计算并保存最终的模式和标签
        self.modes_ = modes
        self.labels_ = self.calculate_labels(modes, curr_h, inter_weights)
