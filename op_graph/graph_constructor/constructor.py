
from utils import logger
from op_graph import OpGraph
from op_graph.op import build_operator
from oneflow.core.job.job_pb2 import Job

class OpGraphConstructor():

    def __init__(self) -> None:
        pass    
        
    def build_op_graph(self, graph_name: str) -> OpGraph:
        oneflow_graph_proto = self._get_graph_proto(graph_name)
        op_graph = self._build_from_graph_proto(oneflow_graph_proto)
        return op_graph
    
    def _get_graph_proto(self, graph_name: str) -> Job:
        # TODO: add graph proto
        raise NotImplementedError

    def _build_from_graph_proto(self, graph_proto: Job) -> OpGraph:
        assert isinstance(graph_proto, Job), f"Ivalid type {type(graph_proto)} for graph_proto"
        op_graph = OpGraph()

        # add operators from graph proto
        for op_conf in graph_proto.net.op:
            name = op_conf.name
            try:
                op_node = build_operator(op_conf)
            except NotImplementedError:
                logger.debug(f"Ignoring operator {name} due to missing implementation.")
                continue
            op_graph.add_node(name)
            op_graph.nodes[name].operator = op_node

        # connect edges
        for u, u_op in op_graph.nodes(data='operator'):
            u_out = set(u_op.output_tensors.values())
            for v, v_op in op_graph.nodes(data='operator'):
                v_in = set(v_op.input_tensors.values())
                used_tensors = list(u_out & v_in)
                if used_tensors:
                    op_graph.add_edge(u, v, used_tensors=used_tensors)
        
        # derive tensor shapes
        #TODO: finish this part

        return op_graph