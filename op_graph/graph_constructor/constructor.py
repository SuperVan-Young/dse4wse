import os
import sys
from abc import ABC, abstractmethod
from typing import Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import logger, TensorInfo
from graph import OpGraph, build_op_graph_from_operator_list
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
        self._postprocess(op_graph)
        return op_graph
    
    @abstractmethod
    def _get_onnx_model(self):
        raise NotImplementedError

    def _build_from_onnx_model(self, onnx_model) -> OpGraph:

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

        # FIXME: inplace should be deprecated in the future, since weight is also produced
        tensor_infos = {val.name: get_tensor_info(val, inplace=False) for val in onnx_model.graph.value_info}
        tensor_infos.update({val.name: get_tensor_info(val, inplace=True) for val in onnx_model.graph.input})
        tensor_infos.update({val.name: get_tensor_info(val, inplace=True) for val in onnx_model.graph.output})
        tensor_infos.update({val.name: get_tensor_info_from_initializer(val, inplace=True) for val in onnx_model.graph.initializer})

        # add operators from graph proto
        operators = []
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
            operators.append(op_node)

        op_graph = build_op_graph_from_operator_list(operators)

        logger.info(f"Summary for building op graph:")
        logger.info(f"Number of Op in original ONNX model: {len(onnx_model.graph.node)}")
        logger.info(f"Number of Op in original exported Op Graph: {op_graph.number_of_nodes()}")
        
        op_graph._onnx_model = onnx_model

        return op_graph
    
    @abstractmethod
    def _postprocess(self, op_graph: OpGraph):
        raise NotImplementedError