import copy
import random
from collections import deque


#%%  KEY：计算Euler图的圈分解
"""
input.vectors：G：nx.graph对象，待分解的Euler图
ouput：list of list对象，分解的顶点集合划分
"""
def GreedyCycleDecomposition(G):

    ##  数据预处理
    graph_origin = {node: list(neighbors) for node, neighbors in G.adjacency()}
    deg_origin = {node: len(neighbors) for node, neighbors in graph_origin.items()}  # 初始化度数字典

    ##  检查图是否是欧拉图：要求所有顶点度数为偶数
    if any(d % 2 != 0 for d in deg_origin.values()):
        raise ValueError("图不是欧拉图")

    ##  主循环：直到图中没有边
    decomposition = []  # 存储分解的圈
    graph=copy.deepcopy(graph_origin)
    deg=copy.deepcopy(deg_origin)
    while any(d > 0 for d in deg.values()):

        # 随机选择一个度数大于0的顶点
        available_nodes = [node for node, d in deg.items() if d > 0]
        if not available_nodes:
            break
        v = random.choice(available_nodes)

        # 使用BFS找到包含v的最小圈
        cycle = find_cycle_bfs(v, graph)
        decomposition.append(cycle)

        # 移除圈中的所有边
        n = len(cycle)
        for i in range(n - 1):
            u1, u2 = cycle[i], cycle[i + 1]

            # 从邻接表中移除边
            if u2 in graph[u1]:
                graph[u1].remove(u2)
            if u1 in graph[u2]:
                graph[u2].remove(u1)

            # 更新度数
            deg[u1] -= 1
            deg[u2] -= 1

    return decomposition


#%%  KEY：使用BFS从起点start出发寻找最小圈。
"""
start: int对象，起始顶点
graph: dict对象，当前图的邻接表表示
output：None or list对象，圈节点序列，若找不到圈则返回None。
"""
def find_cycle_bfs(start, graph):
    # 初始化数据结构
    parent = {start: None}
    visited = {start}
    queue = deque([start])

    while queue:
        u = queue.popleft()
        # 遍历当前节点的所有邻居
        for w in graph[u]:
            # 如果邻居节点未被访问
            if w not in visited:
                visited.add(w)
                parent[w] = u
                queue.append(w)
            # 如果遇到已访问的非父节点（回边）
            elif w != parent[u]:
                # 获取u和w的最近公共祖先(LCA)
                ancestors = set()
                cur = u
                while cur is not None:
                    ancestors.add(cur)
                    cur = parent[cur]

                # 从w向上回溯寻找LCA
                cur = w
                while cur not in ancestors:
                    cur = parent[cur]
                lca = cur

                # 构建从u到LCA的路径
                path_u = []
                cur = u
                while cur != lca:
                    path_u.append(cur)
                    cur = parent[cur]
                path_u.append(lca)

                # 构建从w到LCA的路径
                path_w = []
                cur = w
                while cur != lca:
                    path_w.append(cur)
                    cur = parent[cur]

                # 组合形成完整的圈
                cycle = path_u + path_w[::-1] + [path_u[0]]
                return cycle

    return None