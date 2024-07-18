from numba import jit
import numpy as np


# 计算最小和最大平方范数
@jit(nopython=True)
def _find_min_max_l2_impl(points):
    # 初始计算点集中第一对点的距离平方
    x_sq = 0
    for d in range(points.shape[1]):
        x_sq += (points[0, d] - points[1, d]) ** 2

    l2_min = x_sq
    l2_max = x_sq

    # 遍历点集中的每一对点，计算平方范数，并更新最小和最大距离平方
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x_sq = 0
            for d in range(points.shape[1]):
                x_sq += (points[i, d] - points[j, d]) ** 2

            if x_sq == 0:
                continue

            if x_sq < l2_min:
                l2_min = x_sq
            if x_sq > l2_max:
                l2_max = x_sq

    return l2_min, l2_max


# 计算两个矩阵中向量之间的平方范数
@jit("void(float32[:,:], float32[:,:], float32[:,:])", nopython=True)
def _l2_squared_impl(left, right, result):
    for i in range(len(left)):
        for j in range(len(right)):
            result[i, j] = np.sum((left[i] - right[j]) ** 2)


# 计算单个矩阵中所有向量之间的平方范数
@jit("void(float32[:,:], float32[:,:])", nopython=True)
def _l2_squared_self_impl(left, result):
    for i in range(len(left)):
        for j in range(i, len(left)):
            r = np.sum((left[i] - left[j]) ** 2)
            result[i, j] = r
            result[j, i] = r


# 调用 _l2_squared_self_impl 和 _l2_squared_impl 来计算两个矩阵中向量之间的平方范数，
# 如果没有提供 right 参数，则计算单个矩阵中所有向量之间的距离平方；如果提供了 right 参数，则计算两个矩阵中向量之间的距离平方。
def numba_l2_squared(left, right=None):
    assert len(left.shape) == 2

    if right is None:
        result = np.empty((len(left),) * 2, dtype=np.float32)
        _l2_squared_self_impl(left.astype(np.float32), result)
    else:
        assert len(right.shape) == 2
        assert right.shape[1] == left.shape[1]
        result = np.empty((len(left), len(right)), dtype=np.float32)
        _l2_squared_impl(left.astype(np.float32), right.astype(np.float32), result)
    return result
