import numpy as np


# 用于处理约束数组，将其转换为一个对称且唯一的约束列表
def constraint_list_from_constraints(constraints):
    # 转化为numpy数组且不进行复制
    constraints = np.array(constraints, copy=False)
    # 验证约束数组是否为二维数组，且第二维的大小为2，并且元素类型为整数。如果不符合条件，抛出 ValueError
    if len(constraints.shape) == 2 and constraints.shape[1] == 2 and np.issubdtype(constraints.dtype, np.integer):
        pass
    else:
        raise ValueError("Constraints not understood, shape={}, dtype={}".format(constraints.shape, constraints.dtype))

    # 生成对称约束
    cl = np.concatenate([constraints, constraints[:, ::-1]])
    # 去重
    return np.unique(cl, axis=0)


# Taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
# 创建必须链接(ml)和禁止链接(cl)约束图，并生成约束图的连通组件
def preprocess_constraints(ml, cl, n):
    # 检查所有的必须链接和禁止链接的索引是否在有效范围内（0 到 n-1）。
    assert np.all(np.logical_and(ml >= 0, ml < n))
    assert np.all(np.logical_and(cl >= 0, cl < n))

    # 使用字典表示邻接表，其中每个点对应一个集合，用于存储相邻的点。
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    # 将必须链接和禁止链接分别添加到对应的邻接表中
    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)
    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)
    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    # See http://www.techiedelight.com/transitive-closure-graph/ for more details
    # 使用DFS生成必须链接的连通组件，并将每个组件中的所有点两两相连
    visited = [False] * n
    neighborhoods = []
    for i in range(n):
        if not visited[i] and ml_graph[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    # 根据必须链接的连通组件更新禁止链接的邻接表，确保连通组件中的所有点都被正确约束
    for (i, j) in cl:
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)

        for y in ml_graph[j]:
            add_both(cl_graph, i, y)

        for x in ml_graph[i]:
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    # 检查是否存在不一致的约束（即一个点既有必须链接又有禁止链接），如果存在则抛出错误
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise ValueError('Inconsistent constraints between {} and {}'.format(i, j))

    return ml_graph, cl_graph, neighborhoods


# 生成不能连接约束的传递闭包
def transitive_closure_constraints(cl_constraints, ml_constraints, n):
    # 得到必须连接约束图和不可连接约束图，字典结构，其中键是节点，值是与键节点有约束关系的节点列表
    ml_graph, cl_graph, _ = preprocess_constraints(ml_constraints, cl_constraints, n)
    # 初始化空列表
    result = []
    # 遍历不可连接约束图cl_graph，将所有约束对(k, v)添加到result列表中
    for k, vs in cl_graph.items():
        for v in vs:
            result.append((k, v))
    # 将结果列表转换为numpy数组
    return np.array(result, dtype=np.int32)
