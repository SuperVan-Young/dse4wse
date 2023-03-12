import networkx as nx

from networkx import DiGraph

class OpGraph(DiGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self._onnx_graph = None