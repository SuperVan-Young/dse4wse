import torch
import onnx
import os
import sys
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import logger, TensorInfo
from graph import OpGraph
from op.build_op import build_operator

MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
if not os.path.exists(MODEL_CACHE_DIR):
    os.mkdir(MODEL_CACHE_DIR)

class OpGraphConstructor(ABC):
    """Build OpGraph from ONNX models
    """
    def __init__(self, *args, **kwargs) -> None:
        pass

    def build_op_graph(self) -> OpGraph:
        onnx_model = self._get_onnx_model()
        op_graph = self._build_from_onnx_model(onnx_model)
        op_graph.name = self._graph_name
        return op_graph
    
    @property
    @abstractmethod
    def _graph_name(self):
        return None
    
    @abstractmethod
    def _get_onnx_model(self):
        raise NotImplementedError

    def _build_from_onnx_model(self, onnx_model) -> OpGraph:
        op_graph = OpGraph()

        def get_tensor_info(val, inplace):
            name = val.name
            shape = [d.dim_value for d in val.type.tensor_type.shape.dim]
            dtype = val.type.tensor_type.elem_type
            tensor_info = TensorInfo(shape=shape, onnx_dtype=dtype, name=name, inplace=inplace)
            return tensor_info
        
        def get_tensor_info_from_initializer(val, inplace):
            name = val.name
            shape = [d for d in val.dims]
            dtype = val.data_type
            tensor_info = TensorInfo(shape=shape, onnx_dtype=dtype, name=name, inplace=inplace)
            return tensor_info

        tensor_infos = {val.name: get_tensor_info(val, inplace=False) for val in onnx_model.graph.value_info}
        tensor_infos.update({val.name: get_tensor_info(val, inplace=True) for val in onnx_model.graph.input})
        tensor_infos.update({val.name: get_tensor_info(val, inplace=True) for val in onnx_model.graph.output})
        tensor_infos.update({val.name: get_tensor_info_from_initializer(val, inplace=True) for val in onnx_model.graph.initializer})

        # add operators from graph proto
        for op_proto in onnx_model.graph.node:
            name = op_proto.name
            op_type = op_proto.op_type
            input_tensors = [tensor_infos[i] for i in op_proto.input]
            output_tensors = [tensor_infos[i] for i in op_proto.output]
            try:
                op_node = build_operator(name, op_type, input_tensors, output_tensors)
            except NotImplementedError:
                logger.info(f"Ignoring operator {name} due to missing implementation.")
                continue
            op_graph.add_node(name)
            op_graph.nodes[name]['operator'] = op_node

        # connect edges
        for u, u_op in op_graph.nodes(data='operator'):
            u_out = set(u_op.output_tensors.values())
            for v, v_op in op_graph.nodes(data='operator'):
                v_in = set(v_op.input_tensors.values())
                used_tensors = list(u_out & v_in)
                if used_tensors:
                    op_graph.add_edge(u, v, used_tensors=used_tensors)

        logger.info(f"Summary for building op graph:")
        logger.info(f"Number of Op in original ONNX model: {len(onnx_model.graph.node)}")
        logger.info(f"Number of Op in original exported Op Graph: {op_graph.number_of_nodes()}")
        
        op_graph._onnx_model = onnx_model

        return op_graph