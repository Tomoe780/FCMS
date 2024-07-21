import matplotlib.pyplot as plt
import numpy as np


def cms_synth(x, cl, ml, use_cuda=True):
    from CMS import CMS, AutoLinearPolicy, constraint_list_from_constraints, transitive_closure_constraints

    # 计算不可链接约束和必须链接约束的传递闭包
    cl = transitive_closure_constraints(cl, ml, len(x))
    cl = constraint_list_from_constraints(cl)

    # 创建自动线性带宽策略
    iterations = 50
    pol = AutoLinearPolicy(x, iterations)

    # 初始化CMS算法
    cms = CMS(pol, max_iterations=iterations, blurring=False, kernel=.2,
              use_cuda=use_cuda, label_merge_b=.0, label_merge_k=.995)

    # 训练模型并预测聚类结果
    cms.fit(x, cl)

    # 可视化
    from CMS.Plotting import plot_clustering
    plot_clustering(x, cms.labels_, cms.modes_, cl=None)
    plt.show()
    return cms.labels_


# 定义了一个字典，该字典包含不同数据集的加载方法。这些方法从文件系统中加载文本数据或生成合成数据集
def get_datasets():
    from Util.Datasets import load_text_data, load_moons, load_s4
    import os

    res_path = os.path.join(os.path.dirname(__file__), 'data')

    # 返回一个字典，其中每个键是数据集的名称，每个值是一个匿名函数（lambda函数）或直接调用的函数，用于加载相应的数据集
    return {
        'aggregation': lambda: load_text_data(os.path.join(res_path, 'Aggregation.txt'), 'aggregation'),
        'moons': lambda: load_moons(500),
        'jain': lambda: load_text_data(os.path.join(res_path, 'jain.txt'), 'jain'),
        's4': lambda: load_s4(),
    }


def main():
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from Util.Sampling import generate_constraints_fixed_count
    from Util.CsvWriter import CsvWriter
    from argparse import ArgumentParser

    datas = get_datasets()

    # 解析命令行参数
    parser = ArgumentParser()
    parser.add_argument('--data', choices=datas.keys(), default='s4')
    parser.add_argument('--repeats', metavar='N', type=int, default=1)
    parser.add_argument('--constraint-factor', metavar='F', type=float, default=1.)
    parser.add_argument('--nocuda', action="store_false", dest='use_cuda')
    args = parser.parse_args()

    file_name = 'cms-synth-{}.csv'.format(args.data)

    use_cuda = args.use_cuda

    # 使用CsvWriter 打开一个CSV文件，并在每次运行中写入聚类结果
    with CsvWriter(file_name) as writer:
        # 循环运行指定次数
        for run in range(args.repeats):
            print('Run {}/{}'.format(run + 1, args.repeats))

            try:
                # 加载并预处理数据集
                x, y = datas[args.data]().normalized_linear().train
            except Exception as ex:
                raise RuntimeError("Failed to load dataset：{}".format(args.data)) from ex

            # 可视化数据集
            from CMS.Plotting import plot_clustering
            plot_clustering(x, y)
            plt.show()

            # 生成固定数量的约束
            n_c = int(len(y) * args.constraint_factor)
            cl, ml = generate_constraints_fixed_count(y, n_c)

            # 执行CMS聚类算法并预测结果
            y_pred = cms_synth(np.copy(x), np.copy(cl), np.copy(ml), use_cuda=use_cuda)

            # 计算ARI和NMI评分
            ari = adjusted_rand_score(y, y_pred)
            nmi = normalized_mutual_info_score(y, y_pred)
            # 将结果写入CSV文件
            writer.write_row(algo='cms', data=args.data, ari=ari, nmi=nmi, n_c=n_c)


if __name__ == '__main__':
    main()
