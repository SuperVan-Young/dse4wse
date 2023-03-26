
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itertools import chain

from graph import OpGraph, build_op_graph_from_operator_list
from op import MatMulOperator, Operator
from utils import TensorInfo, TrainingConfig, ArchConfig, logger

BFLOAT16 = 10

MATMUL_BLOCK_SIZE = 64  # otherwise we cannot cover SRAM reference bandwidth

class AttentionModule():

    def __init__(self,
                 attention_heads: int,
                 hidden_size: int,
                 sequence_length: int,
                 micro_batch_size: int,
                 mini_batch_size: int,
                 number_of_layers: int,  # total
                 data_parallel_size: int,
                 model_parallel_size: int,
                 tensor_parallel_size: int,
                 swap_weight_timesteps: int,
                 ) -> None:
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.micro_batch_size = micro_batch_size
        self.mini_batch_size = mini_batch_size
        self.number_of_layers = number_of_layers
        self.data_parallel_size = data_parallel_size
        self.model_parallel_size = model_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        assert number_of_layers % model_parallel_size == 0
        self.layer_of_pipeline_stage = number_of_layers // model_parallel_size
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
    
    def alloc_core_and_derive_sbp_sig(self, arch_config: ArchConfig):
        """Naive sbp derivation for the operation.

        core allocation: each operator use all cores
        sbp derivation: no consideration on intra_sbp_sig boxing cost
        """
        for name, op in self._op_graph.nodes(data='operator'):
            op:Operator
            op.num_core_range = list(range(1, arch_config.get_total_cores() + 1))
            
            op.debug = True
            logger.info(f"Derive sbp for operator {name}:")
            for tensor_name, tensor in chain(op.input_tensors.items(), op.output_tensors.items()):
                logger.info(f"- {tensor_name}: {tensor.shape}")

            op.generate_candidate_intra_sbp_sigs()
            intra_sbp_sigs, inter_sbp_sigs = op.find_best_sbp_signature(arch_config, self._training_config, {})
            op.final_intra_sbp_sigs = intra_sbp_sigs
            op.final_inter_sbp_sigs = inter_sbp_sigs

    def get_training_throughput(self, arch_config: ArchConfig):
        """ in terms of sequence per second
        """
        logger.info("Calculating throughput of attention module")

        single_stage_fp_latency = self.get_fp_latency(arch_config)
        single_stage_bp_latency = self.get_bp_latency(arch_config)
        pipeline_factor = self.number_of_layers + (self.mini_batch_size // self.micro_batch_size - 1)
        whole_batch_latency = pipeline_factor * (single_stage_bp_latency + single_stage_fp_latency)  # cycles
        whole_batch_time = whole_batch_latency / arch_config.get_core_frequency()                    # seconds
        sequence_per_second = self.mini_batch_size / whole_batch_time
        return sequence_per_second

    def get_fp_latency(self, arch_config: ArchConfig):
        inter_wafer_bandwidth = arch_config.get_interconnect_bandwidth('wafer')

        # fetch activation from previous wafer for data parallel times
        input_tensors = self._op_graph.get_tensors(kind=['input'])
        input_tensor_total_size = sum([tensor.size() for tensor in input_tensors.values()]) * self.data_parallel_size
        input_cross_wafer_latency = input_tensor_total_size / inter_wafer_bandwidth

        # compute for #swap_weight_timesteps
        op_graph_fp_latency = self._op_graph.get_propagation_latency(arch_config, forward=True) * self.swap_weight_timesteps * self.layer_of_pipeline_stage

        # load weight from DRAM once and all-gather for all data-parallel 'workers'
        swap_weight_latency = self._swap_weight_latency(arch_config) * self.layer_of_pipeline_stage

        # compute and swap weight can be overlapped
        # comm cost for output on same device is ignored 
        total_latency = input_cross_wafer_latency + max(op_graph_fp_latency, swap_weight_latency)

        logger.info("Summary for forward propagation latency (latency with same stage number can be overlapped)")
        logger.info(f"- input cross wafer latency (1): {int(input_cross_wafer_latency):>15d} cycles ({input_cross_wafer_latency/total_latency:.2%})")
        logger.info(f"- op graph fp latency       (2): {int(op_graph_fp_latency):>15d} cycles ({op_graph_fp_latency/total_latency:.2%})")
        logger.info(f"- swap weight latency       (2): {int(swap_weight_latency):>15d} cycles ({swap_weight_latency/total_latency:.2%})")

        return total_latency
    
    def get_bp_latency(self, arch_config: ArchConfig):
        reticle_cluster_boundary_size = max(arch_config.get_array_size('reticle_height'), arch_config.get_array_size('reticle_width'))
        inter_reticle_bandwidth = arch_config.get_interconnect_bandwidth('reticle')
        inter_worker_cluster_bandwidth = reticle_cluster_boundary_size * inter_reticle_bandwidth
        inter_wafer_bandwidth = arch_config.get_interconnect_bandwidth('wafer')
        dram_bandwidth = arch_config.get_wafer_dram_bandwidth()

        # fetch grad from successive wafer once and broadcast to all data parallel 'worker'
        grad_tensors = self._op_graph.get_tensors(kind=['output'])
        grad_tensors_total_size = sum([tensor.size() for tensor in grad_tensors.values()])
        grad_cross_wafer_latency = grad_tensors_total_size / inter_wafer_bandwidth
        grad_broadcast_latency = (self.data_parallel_size - 1) * grad_tensors_total_size / inter_worker_cluster_bandwidth
        grad_total_latency = grad_cross_wafer_latency + grad_broadcast_latency

        # rematerialize activation, similar to fp
        assert self._training_config.activation_checkpoint == {}, "Currently not support fine granularity rematerialization"
        input_tensors = self._op_graph.get_tensors(kind=['input'])
        input_tensor_total_size = sum([tensor.size() for tensor in input_tensors.values()]) * self.data_parallel_size
        input_dram_latency = input_tensor_total_size / dram_bandwidth

        op_graph_fp_latency = self._op_graph.get_propagation_latency(arch_config, forward=True) * self.swap_weight_timesteps * self.layer_of_pipeline_stage
        swap_weight_latency = self._swap_weight_latency(arch_config) * self.layer_of_pipeline_stage

        rematerialization_total_latency = input_dram_latency + max(op_graph_fp_latency, swap_weight_latency)

        # compute for # swap_weight_timesteps
        op_graph_bp_latency = self._op_graph.get_propagation_latency(arch_config, forward=False) * self.swap_weight_timesteps * self.layer_of_pipeline_stage

        # rematerilization and fetch grad can be overlapped
        # compute and swap weight can be overlapped
        total_latency = max(grad_total_latency, rematerialization_total_latency) + max(op_graph_bp_latency, swap_weight_latency)

        logger.info("Summary for backward propagation latency (latency with same stage number can be overlapped)")
        logger.info(f"- grad transmission latency (1): {int(grad_total_latency):>15d} cycles ({grad_total_latency/total_latency:.2%})")
        logger.info(f"- rematerialization latency (1): {int(rematerialization_total_latency):>15d} cycles ({rematerialization_total_latency/total_latency:.2%})")
        logger.info(f"- op graph bp latency       (2): {int(op_graph_bp_latency):>15d} cycles ({op_graph_bp_latency/total_latency:.2%})")
        logger.info(f"- swap weight latency       (2): {int(swap_weight_latency):>15d} cycles ({swap_weight_latency/total_latency:.2%})")
    
        return total_latency

    def _swap_weight_latency(self, arch_config: ArchConfig):
        
        dram_bandwidth = arch_config.get_wafer_dram_bandwidth()

        weight_tensors = self._op_graph.get_tensors(kind=['weight'])
        weight_tensor_total_size = sum([tensor.size() for tensor in weight_tensors.values()])
        weight_dram_aceess_latency = weight_tensor_total_size / dram_bandwidth

        reticle_cluster_boundary_size = max(arch_config.get_array_size('reticle_height'), arch_config.get_array_size('reticle_width'))
        inter_reticle_bandwidth = arch_config.get_interconnect_bandwidth('reticle')
        inter_worker_cluster_bandwidth = reticle_cluster_boundary_size * inter_reticle_bandwidth

        all_gather_round = (self.swap_weight_timesteps + self.data_parallel_size - 1) // self.data_parallel_size
        weight_all_gather_size_per_round = weight_tensor_total_size / self.swap_weight_timesteps * (self.data_parallel_size - 1)
        weight_all_gather_total_size = weight_all_gather_size_per_round * all_gather_round 
        weight_all_gather_latency = weight_all_gather_total_size / inter_worker_cluster_bandwidth

        return weight_dram_aceess_latency + weight_all_gather_latency
    
    
