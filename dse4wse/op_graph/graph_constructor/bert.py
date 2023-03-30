import onnx
import torch
import os
import sys
from typing import Dict

from utils import logger
from transformers import BertModel, BertConfig
from onnxsim import simplify

from .constructor import MODEL_CACHE_DIR, OpGraphConstructor
from dse4wse.op_graph.graph import OpGraph

# from pretrained model
BERT_CONFIG = BertConfig(**{
    "_name_or_path": "bert-base-uncased",
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": None,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 1,  # originally 12
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.26.1",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 30522
})

class BertOpGraphConstructor(OpGraphConstructor):

    def __init__(self, batch_size=1, max_seq_len=512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def _get_onnx_model(self):
        model_path = os.path.join(MODEL_CACHE_DIR, f"bert_b{self.batch_size}_l{self.max_seq_len}.onnx")

        if not os.path.exists(model_path):
            logger.info(f"Building model {model_path} from scratch")
            
            model = BertModel(config=BERT_CONFIG)
            batch_size, max_seq_len = self.batch_size, self.max_seq_len
            inputs = {
                'input_ids': torch.ones((batch_size, max_seq_len)).long(),
                'attention_mask': torch.ones((batch_size, max_seq_len)).long(),
                'token_type_ids': torch.ones((batch_size, max_seq_len)).long(),
                'position_ids': torch.ones((1, max_seq_len)).long(),
            }
            torch.onnx.export(model, args=(inputs,), f=model_path, export_params=False, training=torch.onnx.TrainingMode.TRAINING)

            logger.info(f"Simplifying model {model_path}")
            
            onnx_model = onnx.load(model_path, load_external_data=False)
            onnx_model, check = simplify(onnx_model, skip_constant_folding=True)
            assert check, f"Failed to simplify onnx model {model_path}"
            onnx.save(onnx_model, model_path)

        else:
            onnx_model = onnx.load(model_path, load_external_data=False)

        return onnx_model
    
    def _postprocess(self, op_graph: OpGraph):
        op_graph.name = "Bert batch=%s seq=%s" % (self.batch_size, self.max_seq_len)

        def get_duplication(name):
            NUM_HIDDEN_LAYER = 12
            return NUM_HIDDEN_LAYER if "/encoder/layer" in name else 1
        op_graph._duplication_table = {name: get_duplication(name) for name in op_graph.nodes()}
    

if __name__ == "__main__":
    constructor = BertOpGraphConstructor()
    constructor.build_op_graph()