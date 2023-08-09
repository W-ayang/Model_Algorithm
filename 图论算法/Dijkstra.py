import matplotlib.pyplot as plt
import networkx as nx
import matplotlib 
#创建有向图
graph = nx.DiGraph()
#创建下标为0开始的6个节点
graph.add_nodes_from(range(0, 6))
#输入带权边的数据
edges = [(0, 1, 2), (0, 2, 3), (0, 4, 7), (0, 5, 2),
         (2, 3, 5), (2, 4, 1),
         (4, 3, 3), (4, 5, 2)]
#输入边
graph.add_weighted_edges_from(edges)
#求解
sd = []
sd_path = []
for i in range(0, 6):
    sd.append(nx.dijkstra_path_length(graph, 0, i, weight='weight'))
    sd_path.append(nx.dijkstra_path(graph, 0, i, weight='weight'))
#输出结果
print("最短路径值为：", sd)
print("最短路径为：", sd_path)
nx.draw(graph, with_labels=True)
plt.show()


