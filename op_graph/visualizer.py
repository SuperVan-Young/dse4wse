
import networkx as nx
import matplotlib.pyplot as plt
from graph import OpGraph

def visualize_op_graph(op_graph: OpGraph):
    for node in op_graph.nodes():
        op_graph.nodes[node]['_vis_depth'] = 0

    for node in nx.topological_sort(op_graph):
        max_vis_depth = max([op_graph.nodes[pred]['_vis_depth']
            for pred in op_graph.predecessors(node)
        ], default=0) + 1
        op_graph.nodes[node]['_vis_depth'] = max_vis_depth

    pos = nx.multipartite_layout(op_graph, '_vis_depth', align='vertical')
    nx.draw_networkx(op_graph, pos=pos, node_size=1, font_size=1)

    plt.savefig('test.pdf')
    plt.clf()