from typing import Union, Dict, Tuple
import torch
import onnx
import os
import sys
import numpy as np
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import logger, TensorInfo
from op_graph import OpGraph
from build_op import build_operator

from onnxsim import simplify
from transformers import BertTokenizer, BertModel

MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
TMP_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
if not os.path.exists(MODEL_CACHE_DIR):
    os.mkdir(MODEL_CACHE_DIR)
if not os.path.exists(TMP_CACHE_DIR):
    os.mkdir(TMP_CACHE_DIR)

class OpGraphConstructor():
    """Build OpGraph from ONNX models
    """

    def __init__(self) -> None:
        pass    
        
    def build_op_graph(self, model_name: str) -> OpGraph:
        onnx_model = self._get_onnx_model(model_name)
        op_graph = self._build_from_onnx_model(onnx_model)
        return op_graph
    
    def _get_onnx_model(self, model_name: str):
        model_path = os.path.join(MODEL_CACHE_DIR, f"{model_name}.onnx")
        model_2_build_func = {
            'bert_model': build_bert_model,
        }

        if not os.path.exists(model_path):
            try:
                logger.info("Building onnx model %s from scratch" % (model_name,))
                build_func = model_2_build_func[model_name]
                build_func()
            except KeyError:
                logger.critical("We don't have onnx model %s" % (model_name, ))
                exit(1)

        logger.debug("Loading model %s" % (model_name,))
        with open(model_path, 'rb') as f:
            model = onnx.load(f)

        # FIXME: onnxsim cannot behave properly on dynamic inputs
        # FIXME: simplification takes too much time
        logger.debug("Simplifying model %s" % (model_name,))
        model, check = simplify(model)
        assert check
        
        # FIXME: This piece of code should only be used for debugging
        tmp_model_path = os.path.join(TMP_CACHE_DIR, f"{model_name}.onnx")
        tmp_weight_path = os.path.join(TMP_CACHE_DIR, f"{model_name}_weight.onnx")
        logger.debug("Saving simplified model to tmp path %s" % (tmp_model_path, ))
        onnx.save(model, tmp_model_path, save_as_external_data=True, location=tmp_weight_path)
        logger.debug("Remove redundant model weight file %s" % (tmp_weight_path, ))
        os.remove(tmp_weight_path)

        return model


    def _build_from_onnx_model(self, onnx_model) -> OpGraph:
        op_graph = OpGraph()

        def get_tensor_info(val):
            name = val.name
            shape = [d.dim_value for d in val.type.tensor_type.shape.dim]
            dtype = val.type.tensor_type.elem_type
            tensor_info = TensorInfo(shape=shape, onnx_dtype=dtype, name=name)
            return tensor_info
        
        def get_tensor_info_from_initializer(val):
            name = val.name
            shape = [d for d in val.dims]
            dtype = val.data_type
            tensor_info = TensorInfo(shape=shape, onnx_dtype=dtype, name=name)
            return tensor_info

        tensor_infos = {val.name: get_tensor_info(val) for val in onnx_model.graph.value_info}
        tensor_infos.update({val.name: get_tensor_info(val) for val in onnx_model.graph.input})
        tensor_infos.update({val.name: get_tensor_info(val) for val in onnx_model.graph.output})
        tensor_infos.update({val.name: get_tensor_info_from_initializer(val) for val in onnx_model.graph.initializer})

        # add operators from graph proto
        for op_proto in onnx_model.graph.node:
            name = op_proto.name
            op_type = op_proto.op_type
            input_tensors = [tensor_infos[i] for i in op_proto.input]
            output_tensors = [tensor_infos[i] for i in op_proto.output]
            try:
                op_node = build_operator(name, op_type, input_tensors, output_tensors)
            except NotImplementedError:
                logger.debug(f"Ignoring operator {name} due to missing implementation.")
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
        
        op_graph._onnx_model = onnx_model

        return op_graph
    

def build_bert_model():
    """Bert model from huggingface.
    Preliminaries: `pip install transformers`
    """
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    batch_size = 1
    max_seq_len = 512

    encoded_input = {
        'input_ids': torch.ones((batch_size, max_seq_len)).long(),
        'attention_mask': torch.ones((batch_size, max_seq_len)).long(),
        'token_type_ids': torch.ones((batch_size, max_seq_len)).long(),
        'position_ids': torch.ones((1, max_seq_len)).long(),
    }

    with open(os.path.join(MODEL_CACHE_DIR, "bert_model.onnx"), 'wb') as f:
        torch.onnx.export(model, args=(encoded_input,), f=f,
                          input_names = ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids'],
                         )

if __name__ == "__main__":
    constructor = OpGraphConstructor()
    constructor.build_op_graph("bert_model")