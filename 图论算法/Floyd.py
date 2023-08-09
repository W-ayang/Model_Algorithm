import networkx as nx
import matplotlib.pyplot as plt
#创建无向图
graph = nx.Graph()
#创建以下标为1开始的5个顶点
graph.add_nodes_from(range(1, 6))
#输入带权边的数据
edges = [(1, 2, 8), (1, 3, 4), (1, 5, 2),
         (2, 3, 4),
         (3, 4, 2), (3, 5, 1),
         (4, 5, 5)]
#输入边
graph.add_weighted_edges_from(edges)
#求解
min_tree = nx.minimum_spanning_tree(graph, weight='weight', algorithm='kruskal')
min_tree_dict = nx.get_edge_attributes(min_tree, 'weight')
tree_size = sum(min_tree_dict.values())
#输出结果
print("最小生成树为：", min_tree_dict)
print("最小生成树大小为：", tree_size)
plt.show()