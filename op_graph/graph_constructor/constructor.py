import torch
import onnx
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import logger
from op_graph import OpGraph
from op_graph.op import build_operator

from onnx.shape_inference import infer_shapes
from onnxsim import simplify
from transformers import BertTokenizer, BertModel

MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
if not os.path.exists(MODEL_CACHE_DIR):
    os.mkdir(MODEL_CACHE_DIR)

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
                build_func = model_2_build_func[model_name]
                build_func()
            except KeyError:
                logger.critical("We don't have onnx model %s" % (model_name, ))
                exit(1)

        with open(model_path, 'rb') as f:
            model = onnx.load(f)
        model, check = simplify(model)
        assert check, "Failed to simplify model %s" % (model_name,)
        model = infer_shapes(model)
        return model


    def _build_from_onnx_model(self, onnx_model) -> OpGraph:
        op_graph = OpGraph()

        # add operators from graph proto
        for op_proto in onnx_model.graph.node:
            name = op_proto.name
            try:
                op_node = build_operator(op_proto)
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
        
        op_graph._onnx_model = onnx_model

        return op_graph
    

def build_bert_model():
    """Bert model from huggingface.
    Preliminaries: `pip install transformers`
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input_ = {k: v for k, v in encoded_input.items() if isinstance(v, (torch.Tensor))}

    with open(os.path.join(MODEL_CACHE_DIR, "bert_model.onnx"), 'wb') as f:
        torch.onnx.export(model, args=(encoded_input_,), f=f)

if __name__ == "__main__":
    constructor = OpGraphConstructor()
    constructor.build_op_graph("bert_model")