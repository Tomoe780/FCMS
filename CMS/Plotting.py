# 可视化聚类结果,绘制了数据点、聚类中心以及必须连接和不能连接的约束
def plot_clustering(points, assignments, centers=None, cl=None, ml=None, ax=None, labels=None,
                    alpha=1., legend=True, center_size=1., center_marker='o', point_size=1., point_marker='x',
                    palette=None, constraints_on_centers=True, pfx=True, equal_axis_scale=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # 根据 pfx 参数，决定是否启用路径效果。路径效果可以增加标记的可见性
    if pfx:
        import matplotlib.patheffects as path_effects
        path_effects = [path_effects.withStroke(linewidth=2, foreground='black')]
    else:
        path_effects = None

    # 确保 points 和 assignments 的长度相等，并验证 points 的形状为 (N, 2)
    if assignments is None:
        assignments = np.zeros(len(points), dtype=int)
    assert len(points) == len(assignments)
    if len(points.shape) != 2 or points.shape[-1] != 2:
        raise ValueError("Invalid points shape: {}".format(points.shape))

    # 辅助函数 plot_wrap，确保即使只有一个点也能正确绘制
    def plot_wrap(fn, pts, *args, **kwargs):
        if len(pts.shape) == 1:
            pts = np.array([pts])
        return fn(pts[:, 0], pts[:, 1], *args, **kwargs)

    # 计算聚类中心的数量，并确保聚类分配的最大值小于聚类中心的数量
    if centers is not None:
        n_centers = len(centers)
        assert np.max(assignments) < n_centers
    else:
        n_centers = np.max(assignments) + 1

    # 如果未提供调色板，则使用 tab10 颜色映射。获取唯一的聚类标签，并创建一个新的轴对象（如果未提供）
    if palette is None:
        colors = plt.get_cmap('tab10').colors
        palette = [colors[i % 10] for i in range(n_centers)]
    unique = np.unique(assignments)
    if ax is None:
        ax = plt.axes()

    # 对于每个唯一的聚类标签，绘制相应的数据点和聚类中心（如果提供）。应用调色板中的颜色和路径效果
    for u in unique:
        label = labels[u] if labels is not None else None
        p_color = palette[u]
        p_points = points[assignments == u]
        plot_wrap(ax.scatter, p_points, marker=point_marker, s=point_size * rcParams['lines.markersize'] ** 2,
                  color=p_color, label=label, alpha=alpha, path_effects=path_effects)
        if centers is not None:
            plot_wrap(ax.scatter, centers[assignments == u, :], marker=center_marker, color=palette[u],
                      s=center_size * rcParams['lines.markersize'] ** 2, alpha=alpha, path_effects=path_effects)

    # 如果提供了必须连接 (ml) 和不能连接 (cl) 的约束，则绘制这些约束线
    cl_anchors = centers if constraints_on_centers else points
    if ml is not None:
        for i, j in ml:
            plot_wrap(ax.plot, np.stack([cl_anchors[i, :], cl_anchors[j, :]], axis=0), '--', color='tab:blue',
                      zorder=0.5, alpha=alpha, path_effects=path_effects)
    if cl is not None:
        for i, j in cl:
            plot_wrap(ax.plot, np.stack([cl_anchors[i, :], cl_anchors[j, :]], axis=0), '--', color='tab:red',
                      zorder=0.5, alpha=alpha, path_effects=path_effects)

    # 如果提供了标签且需要显示图例，则添加图例。如果需要使X轴和Y轴的比例相等，则调整轴的比例。最后，返回轴对象
    if labels is not None and legend:
        ax.legend()
    if equal_axis_scale:
        # Ensure X and Y axis have equal scale in visualization
        plt.gca().set_aspect('equal', adjustable='box')
    return ax
