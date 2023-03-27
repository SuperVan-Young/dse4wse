
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itertools import chain

from graph import OpGraph, build_op_graph_from_operator_list
from op import MatMulOperator, Operator
from utils import TensorInfo, TrainingConfig, GpuArchConfig, logger
from math import ceil

BFLOAT16 = 10
        
class AttentionModule():
    """We estimate attention module on GpuArchConfig
    
    From the same aspect, WSE provides these improvements than traditional GPU architecture:
    - higher inter-GPU bandwidth through inter-reticle connections
    - (possibly swapping input through inter-reticle connections to reduce runtime HBM overhead)

    We adopt Megatron's dataflow, and leverages ZeRO-DP to reduce memory footprint
    """

    def __init__(self,
                 attention_heads: int,
                 hidden_size: int,
                 sequence_length: int,
                 number_of_layers: int,  # total
                 micro_batch_size: int,
                 mini_batch_size: int,
                 data_parallel_size: int,
                 model_parallel_size: int,
                 tensor_parallel_size: int,
                 **kwargs,
                 ) -> None:
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        assert hidden_size % attention_heads == 0
        self.sequence_length = sequence_length
        self.number_of_layers = number_of_layers

        self.micro_batch_size = micro_batch_size
        self.mini_batch_size = mini_batch_size
        self.data_parallel_size = data_parallel_size
        self.model_parallel_size = model_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        assert number_of_layers % model_parallel_size == 0
        assert hidden_size % tensor_parallel_size == 0

        self.layer_of_pipeline_stage = number_of_layers // model_parallel_size

        self._op_graph = self._build_op_graph()
        self._training_config = self._init_training_config()

    def _build_op_graph(self) -> OpGraph:
        B, S, H = self.micro_batch_size, self.sequence_length, self.hidden_size
        head = self.attention_heads

        head_ = head // self.tensor_parallel_size
        H_ = H // self.tensor_parallel_size

        # Each GPU performs the following operators
        # ZeRO-DP allgather weight is transparent to GPU, although technically compute granularity is smaller
        X = TensorInfo(
            name='X',
            shape=(B, S, H),
            onnx_dtype=BFLOAT16,
            kind='input',
            inplace=False,
        )
        W_qkv = TensorInfo(
            name='W_qkv',
            shape=(H, 3 * H_),
            onnx_dtype=BFLOAT16,
            kind='weight',
            inplace=True,
        )
        QKV = TensorInfo(
            name='QKV',
            shape=(B, S, 3 * H_),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        Q = TensorInfo(
            name='Q',
            shape=(B, head_, S, H // head),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        K = TensorInfo(
            name='K',
            shape=(B, head_, H // head, S),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        scores = TensorInfo(
            name='scores',
            shape=(B, head_, S, S),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        V = TensorInfo(
            name='V',
            shape=(B, head_, S, H // head),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        Y = TensorInfo(
            name='Y',
            shape=(B, head_, S, H // head),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        Y_reshape = TensorInfo(
            name='Y_reshape',
            shape=(B, S, H_),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        W_proj = TensorInfo(
            name='W_proj',
            shape=(H_, H),
            onnx_dtype=BFLOAT16,
            kind='weight',
            inplace=True,
        )
        Z = TensorInfo(
            name = 'Z',
            shape=(B, S, H),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        X0_mlp = TensorInfo(
            name='X0_mlp',
            shape=(B, S, H),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        X1_mlp = TensorInfo(
            name='X1_mlp',
            shape=(B, S, 4 * H_),
            onnx_dtype=BFLOAT16,
            kind='activation',
            inplace=False,
        )
        X2_mlp = TensorInfo(
            name='X2_mlp',
            shape=(B, S, H),
            onnx_dtype=BFLOAT16,
            kind='output',
            inplace=False,
        )
        W1_mlp = TensorInfo(
            name='W0_mlp',
            shape=(H, 4 * H_),
            onnx_dtype=BFLOAT16,
            kind='weight',
            inplace=True,
        )
        W2_mlp = TensorInfo(
            name='W1_mlp',
            shape=(4 * H_, H),
            onnx_dtype=BFLOAT16,
            kind='weight',
            inplace=True,
        )

        linear_qkv = MatMulOperator(
            name='linear_qkv',
            op_type='Matmul',
            input_tensors={'A': X, 'B': W_qkv},
            output_tensors={'Y': QKV},
        )
        matmul_qk = MatMulOperator(
            name='matmul_qk',
            op_type='Matmul',
            input_tensors={'A': Q, 'B': K},
            output_tensors={'Y': scores},
        )
        matmul_scorev = MatMulOperator(
            name='matmul_scorev',
            op_type='Matmul',
            input_tensors={'A': scores, 'B': V},
            output_tensors={'Y': Y},
        )
        linear_proj = MatMulOperator(
            name='linear_proj',
            op_type='Matmul',
            input_tensors={'A': Y_reshape, 'B': W_proj},
            output_tensors={'Y': Z},
        )
        linear_wlp0 = MatMulOperator(
            name='linear_mlp0',
            op_type='Matmul',
            input_tensors={'A': X0_mlp, 'B': W1_mlp},
            output_tensors={'Y': X1_mlp},
        )
        linear_wlp1 = MatMulOperator(
            name='linear_mlp1',
            op_type='Matmul',
            input_tensors={'A': X1_mlp, 'B': W2_mlp},
            output_tensors={'Y': X2_mlp},
        )

        operators = [
            linear_qkv,
            matmul_qk,
            matmul_scorev,
            linear_proj,
            linear_wlp0,
            linear_wlp1
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
    
    def __get_propagation_latency(self, arch_config: GpuArchConfig, forward=True):
        total_latency = 0

        # compute latency summation
        for op_name, op in self._op_graph.nodes(data='operator'):
            assert op.op_type == "Matmul"
            # For GPU, we assumes that GPU promises 100% computational power for matmul
            mac_count = op.get_fp_mac_count() if forward else op.get_bp_mac_count()
            compute_power = arch_config.compute_power
            total_latency += mac_count / compute_power

        return total_latency
    
    def __swap_weight_latency(self, arch_config: GpuArchConfig, forward: bool):
        """ Swap weight between different data parallel nodes
        """
        inter_node_bandwidth = arch_config.infiniband_bandwidth

        weight_tensors = self._op_graph.get_tensors(kind=['weight'])
        weight_tensor_total_size = sum([tensor.size() for tensor in weight_tensors.values()])
        if forward:
            # allgather weight
            collective_size = (self.data_parallel_size - 1) / self.data_parallel_size * weight_tensor_total_size 
            collective_bandwidth = self.tensor_parallel_size * inter_node_bandwidth
            collective_latency = collective_size / collective_bandwidth
        else:
            # reducescatter grad (could be BF16)
            collective_size = (self.data_parallel_size - 1) / self.data_parallel_size * weight_tensor_total_size 
            collective_bandwidth = self.tensor_parallel_size * inter_node_bandwidth
            collective_latency = collective_size / collective_bandwidth

        return collective_latency
    
    def __activation_collective_latency(self, arch_config: GpuArchConfig, forward: bool):
        if forward:
            output_tensors = self._op_graph.get_tensors(['output'])
            output_tensor_size = sum([tensor.size() for tensor in output_tensors.values()])
            allreduce_output_latency = 2 * (self.tensor_parallel_size - 1) / self.tensor_parallel_size * output_tensor_size / arch_config.nvlink_bandwidth
            return allreduce_output_latency
        else:
            input_tensors = self._op_graph.get_tensors(['input'])
            input_grad_size = sum([tensor.size() for tensor in input_tensors.values()])
            allreduce_input_latency = 2 * (self.tensor_parallel_size - 1) / self.tensor_parallel_size * input_grad_size / arch_config.nvlink_bandwidth
            return allreduce_input_latency

    def get_training_throughput(self, arch_config: GpuArchConfig):
        """ in terms of sequence per second
        """
        logger.info("Calculating throughput of attention module")

        single_stage_fp_latency = self.get_fp_latency(arch_config)
        single_stage_bp_latency = self.get_bp_latency(arch_config)
        pipeline_factor = self.number_of_layers + (self.mini_batch_size // self.micro_batch_size - 1)
        whole_batch_latency = pipeline_factor * (single_stage_bp_latency + single_stage_fp_latency)  # cycles
        sequence_per_second = self.mini_batch_size / whole_batch_latency
        return sequence_per_second

    def get_fp_latency(self, arch_config: GpuArchConfig):
        inter_node_bandwidth = arch_config.infiniband_bandwidth
        inter_gpu_bandwidth = arch_config.nvlink_bandwidth

        # fetch activation from previous pipeline stage
        # each GPU use 1 infiniband for a microbatch's transmission
        input_tensors = self._op_graph.get_tensors(kind=['input'])
        input_tensor_total_size = sum([tensor.size() for tensor in input_tensors.values()])
        input_cross_pipeline_stage_latency = input_tensor_total_size / inter_node_bandwidth

        # fp latency
        op_graph_fp_latency = self.__get_propagation_latency(arch_config, forward=True) * self.layer_of_pipeline_stage
        allreduce_output_latency = self.__activation_collective_latency(arch_config, forward=True) * self.layer_of_pipeline_stage
        fp_swap_weight_latency = self.__swap_weight_latency(arch_config, forward=True) * self.layer_of_pipeline_stage

        total_latency = input_cross_pipeline_stage_latency + op_graph_fp_latency + allreduce_output_latency + fp_swap_weight_latency

        logger.info("Summary for forward propagation latency (latency with same stage number can be overlapped)")
        logger.info(f"- input cross node latency (1): {int(input_cross_pipeline_stage_latency * 1e9):>15d} ns ({input_cross_pipeline_stage_latency/total_latency:.2%})")
        logger.info(f"- op graph fp latency      (2): {int(op_graph_fp_latency * 1e9):>15d} ns ({op_graph_fp_latency/total_latency:.2%})")
        logger.info(f"- fp swap weight latency   (3): {int(fp_swap_weight_latency * 1e9):>15d} ns ({fp_swap_weight_latency/total_latency:.2%})")
        logger.info(f"- allreduce output latency (4): {int(allreduce_output_latency * 1e9):>15d} ns ({allreduce_output_latency/total_latency:.2%})")

        return total_latency
    
    def get_bp_latency(self, arch_config: GpuArchConfig):
        inter_node_bandwidth = arch_config.infiniband_bandwidth
        inter_gpu_bandwidth = arch_config.nvlink_bandwidth

        # fetch grad from next parallel stage
        output_tensors = self._op_graph.get_tensors(kind=['output'])
        grad_tensors_total_size = sum([tensor.size() for tensor in output_tensors.values()])
        grad_cross_pipeline_stage_latency = grad_tensors_total_size / inter_node_bandwidth

        # rematerialize activation (similar to fp), use ZeRO-R
        assert self._training_config.activation_checkpoint == {}, "Currently not support fine granularity rematerialization"
        input_tensors = self._op_graph.get_tensors(kind=['input'])
        input_tensor_total_size = sum([tensor.size() for tensor in input_tensors.values()])
        allgather_input_latency = input_tensor_total_size / inter_gpu_bandwidth
        
        # bp latency
        op_graph_fp_latency = self.__get_propagation_latency(arch_config, forward=True) * self.layer_of_pipeline_stage
        fp_swap_weight_latency = self.__swap_weight_latency(arch_config, forward=True) * self.layer_of_pipeline_stage
        allreduce_output_latency = self.__activation_collective_latency(arch_config, forward=True) * self.layer_of_pipeline_stage

        rematerialization_total_latency = allgather_input_latency + op_graph_fp_latency + allreduce_output_latency + fp_swap_weight_latency

        # bp latency
        op_graph_bp_latency = self.__get_propagation_latency(arch_config, forward=False) * self.layer_of_pipeline_stage
        bp_swap_weight_latency = self.__swap_weight_latency(arch_config, forward=False) * self.layer_of_pipeline_stage
        allreduce_input_latency = self.__activation_collective_latency(arch_config, forward=False) * self.layer_of_pipeline_stage

        # rematerilization and fetch grad can be overlapped
        # compute and swap weight can be overlapped
        total_latency = grad_cross_pipeline_stage_latency + rematerialization_total_latency + op_graph_bp_latency + bp_swap_weight_latency + allreduce_input_latency

        logger.info("Summary for backward propagation latency (latency with same stage number can be overlapped)")
        logger.info(f"- grad transmission latency (1): {int(grad_cross_pipeline_stage_latency * 1e9):>15d} ns ({grad_cross_pipeline_stage_latency/total_latency:.2%})")
        logger.info(f"- rematerialization latency (2): {int(rematerialization_total_latency * 1e9):>15d} ns ({rematerialization_total_latency/total_latency:.2%})")
        logger.info(f"- op graph bp latency       (3): {int(op_graph_bp_latency * 1e9):>15d} ns ({op_graph_bp_latency/total_latency:.2%})")
        logger.info(f"- bp swap weight latency    (4): {int(bp_swap_weight_latency * 1e9):>15d} ns ({bp_swap_weight_latency/total_latency:.2%})")
        logger.info(f"- allreduce input latency   (5): {int(allreduce_input_latency * 1e9):>15d} ns ({allreduce_input_latency/total_latency:.2%})")
    
        return total_latency
    
    def check_hbm_utilization(self, arch_config: GpuArchConfig) -> bool:
        """Check if these GPU could hold the parameters and activations.
        
        """
        total_hbm_size = arch_config.gpu_memory_size * self.data_parallel_size * self.tensor_parallel_size
        
        weight_tensors = self._op_graph.get_tensors(kind=['weight'])
        total_weight_size = sum([tensor.numel() for tensor in weight_tensors.values()]) * (2 + 2 + self._training_config.get_dynamic_optimizer_state_size() + self._training_config.get_dynamic_optimizer_state_size())

        activation_tensors = self._op_graph.get_tensors(kind=['input', 'output', 'activation'])
        total_activation_size = sum([tensor.numel() for tensor in activation_tensors.values()]) * (2 + 2) * self.data_parallel_size * self.tensor_parallel_size

        utilized_size = total_weight_size + total_activation_size

        logger.info("Summary for checking HBM utilization")
        logger.info(f"Util  HBM: {utilized_size/1e9} GB")        
        logger.info(f"Total HBM: {total_hbm_size/1e9} GB")        

        return utilized_size <= total_hbm_size