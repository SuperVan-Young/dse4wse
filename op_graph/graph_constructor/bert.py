import onnx
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import logger
from transformers import BertTokenizer, BertModel
from onnxsim import simplify
from constructor import MODEL_CACHE_DIR, OpGraphConstructor

class BertOpGraphConstructor(OpGraphConstructor):

    def __init__(self, batch_size=1, max_seq_len=512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def _get_onnx_model(self):
        model_path = os.path.join(MODEL_CACHE_DIR, f"bert_b{self.batch_size}_l{self.max_seq_len}.onnx")

        if not os.path.exists(model_path):
            logger.info(f"Building model {model_path} from scratch")
            
            model = BertModel.from_pretrained("bert-base-uncased")
            batch_size, max_seq_len = self.batch_size, self.max_seq_len
            inputs = {
                'input_ids': torch.ones((batch_size, max_seq_len)).long(),
                'attention_mask': torch.ones((batch_size, max_seq_len)).long(),
                'token_type_ids': torch.ones((batch_size, max_seq_len)).long(),
                'position_ids': torch.ones((1, max_seq_len)).long(),
            }
            torch.onnx.export(model, args=(inputs,), f=model_path, export_params=False)

            logger.info(f"Simplifying model {model_path}")
            
            onnx_model = onnx.load(model_path, load_external_data=False)
            onnx_model, check = simplify(onnx_model)
            assert check, f"Failed to simplify onnx model {model_path}"
            onnx.save(onnx_model, model_path)

        else:
            onnx_model = onnx.load(model_path, load_external_data=False)

        return onnx_model

if __name__ == "__main__":
    constructor = BertOpGraphConstructor()
    constructor.build_op_graph()