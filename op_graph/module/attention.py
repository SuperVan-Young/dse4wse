
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itertools import chain

from graph import OpGraph, build_op_graph_from_operator_list
from op import MatMulOperator
from utils import TensorInfo, TrainingConfig, ArchConfig

BFLOAT16 = 10

class AttentionModule():

    def __init__(self,
                 attention_heads: int,
                 hidden_size: int,
                 sequence_length: int,
                 micro_batch_size: int,
                 mini_batch_size: int,
                 number_of_layers: int,  # total
                 data_parallel_size: int,
                 pipeline_model_parallel_size: int,
                 swap_weight_timesteps: int,
                 ) -> None:
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.micro_batch_size = micro_batch_size
        self.mini_batch_size = mini_batch_size
        self.number_of_layers = number_of_layers
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.data_parallel_size = data_parallel_size
        self.swap_weight_timesteps = swap_weight_timesteps

        self._op_graph = self._build_op_graph()
        self._training_config = self._init_training_config()

    def _build_op_graph(self) -> OpGraph:
        B, S, H = self.micro_batch_size, self.sequence_length, self.hidden_size
        T = self.swap_weight_timesteps
        head = self.attention_heads
        assert H % head == 0, f"#hidden_size {H} cannot be evenly split by #attention_head {head}"
        assert head % T == 0, f'#attention_head {head} cannot be evenly split by #swap_weight_timesteps {T}'

        X = TensorInfo(
            name='X',
            shape=(B, S, H),
            onnx_dtype=BFLOAT16,
            kind='input',
            inplace=False,
        )
        W_qkv = TensorInfo(
            name='W_qkv',
            shape=(H, 3 * H // T),
            onnx_dtype=BFLOAT16,
            kind='weight',
            inplace=True,
        )
        QKV = TensorInfo(
            name='QKV',
            shape=(B, S, 3 * H // T),
            onnx_dtype=BFLOAT16,
            kind='input',
            inplace=False,
        )
        Q = TensorInfo(
            name='Q',
            shape=(B, head // T, S, H // head),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        K = TensorInfo(
            name='K',
            shape=(B, head // T, H // head, S),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        scores = TensorInfo(
            name='scores',
            shape=(B, head // T, S, S),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        V = TensorInfo(
            name='V',
            shape=(B, head // T, S, H // head),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        Y = TensorInfo(
            name='Y',
            shape=(B, head // T, S, H // head),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        Y_reshape = TensorInfo(
            name='Y_reshape',
            shape=(B, S, H // T),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        W_proj = TensorInfo(
            name='W_proj',
            shape=(H // T, H),
            onnx_dtype=BFLOAT16,
            kind='weight',
            inplace=True,
        )
        Z = TensorInfo(
            name = 'Z',
            shape=(B, S, H),
            onnx_dtype=BFLOAT16,
            kind='output',
            inplace=False,
        )

        linear_qkv = MatMulOperator(
            name='linear_qkv',
            op_type='Matmul',
            input_tensors={'A': X, 'B': W_qkv},
            output_tensors={'Y': QKV}
        )
        matmul_qk = MatMulOperator(
            name='matmul_qk',
            op_type='Matmul',
            input_tensors={'A': Q, 'B': K},
            output_tensors={'Y': scores}
        )
        matmul_scorev = MatMulOperator(
            name='matmul_scorev',
            op_type='Matmul',
            input_tensors={'A': scores, 'B': V},
            output_tensors={'Y': Y}
        )
        linear_proj = MatMulOperator(
            name='linear_proj',
            op_type='Matmul',
            input_tensors={'A': Y_reshape, 'B': W_proj},
            output_tensors={'Y': Z}
        )
        #TODO: add reshape operators to calculate comm cost

        operators = [
            linear_qkv,
            matmul_qk,
            matmul_scorev,
            linear_proj,
        ]
        op_graph = build_op_graph_from_operator_list(operators)
        return op_graph
    
    def _init_training_config(self):
        config = TrainingConfig(
            optimizer='Adam',
            precision=BFLOAT16,
            mix_precision=True,
            activation_checkpoint={}
        )
        return config

    def get_fp_latency(self, arch_config: ArchConfig):
        # fetch activation from previous wafer
        inter_wafer_bandwidth = arch_config.get_interconnect_bandwidth('wafer')
        input_tensors = self._op_graph.get_tensors(kind=['input'])
        input_tensor_total_size = sum([tensor.size() for tensor in input_tensors.values()])
        input_comm_latency = input_tensor_total_size / inter_wafer_bandwidth

        # compute for #swap_weight_timesteps
        op_graph_fp_latency = self._op_graph.get_fp_latency()
