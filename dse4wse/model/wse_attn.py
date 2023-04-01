
from math import ceil

from typing import Dict
from itertools import product
from functools import reduce
import networkx as nx

from dse4wse.op_graph.graph import OpGraph, build_op_graph_from_operator_list
from dse4wse.op_graph.op import BaseOperator, MatMulOperator
from dse4wse.pe_graph.hardware import WaferScaleEngine, Reticle
from dse4wse.pe_graph.task import (
    ListWaferTask, 
    ThreeStageReticleTaskGenerator,
    FusedReticleTask,
    ComputeReticleTask,
    DramAccessReticleTask,
    PeerAccessReticleTask,
    BaseReticleTask,
)
from dse4wse.pe_graph.mapper import get_default_mapper
from dse4wse.pe_graph.evaluator import LpReticleLevelWseEvaluator
from dse4wse.utils import TensorInfo, TrainingConfig, logger, factoring

BFLOAT16 = 10

class WseTransformerRunner():
    """
    Run transformer layer on WSE.
    """

    def __init__(self,
                 attention_heads: int,
                 hidden_size: int,
                 sequence_length: int,
                 number_of_layers: int,
                 micro_batch_size: int,
                 mini_batch_size: int,
                 data_parallel_size: int,
                 model_parallel_size: int,
                 tensor_parallel_size: int,
                 wafer_scale_engine: WaferScaleEngine,
                 inter_wafer_bandwidth: int,
                 training_config: TrainingConfig,
                 zero_dp_os: bool = True,
                 zero_dp_g: bool = True,
                 zero_dp_p: bool = False,
                 zero_r_pa: bool = True,
                 ) -> None:
        # model parameters
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        assert hidden_size % attention_heads == 0
        self.sequence_length = sequence_length
        self.number_of_layers = number_of_layers

        # parallelism parameters
        self.micro_batch_size = micro_batch_size
        self.mini_batch_size = mini_batch_size
        self.data_parallel_size = data_parallel_size
        self.model_parallel_size = model_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        assert number_of_layers % model_parallel_size == 0
        assert hidden_size % tensor_parallel_size == 0

        # ZeRO settings
        self.zero_dp_os = zero_dp_os
        self.zero_dp_g = zero_dp_g
        self.zero_dp_p = zero_dp_p
        self.zero_r_pa = zero_r_pa

        # wse platform
        self.wafer_scale_engine = wafer_scale_engine
        self.inter_wafer_bandwidth = inter_wafer_bandwidth

        # training settings 
        self.training_config = training_config

        self.num_layer_per_pipeline_stage = number_of_layers // model_parallel_size
        self.num_pipeline_stage_per_wafer = wafer_scale_engine.reticle_array_height * wafer_scale_engine.reticle_array_width // tensor_parallel_size

        self._op_graph = self._build_op_graph()

    def _build_op_graph(self) -> OpGraph:
        """
        Coarse-grained op-graph that one reticle could execute at a time.
        Operators are splitted according to Megatron-LM
        """
        B, S, H = self.micro_batch_size, self.sequence_length, self.hidden_size
        head = self.attention_heads

        head_ = (head // self.tensor_parallel_size)
        H_ = (H // self.tensor_parallel_size)
        if self.zero_dp_p:
            head_ = ceil(head_ / self.data_parallel_size)
            H_ = ceil(H_ / self.data_parallel_size)
            

        # reuse op_graph data structure to record, which is unnecessary...
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
    
    def _get_compute_latency(self, forward: bool) -> float:
        """ Assume 100% compute resource utilization
        (propagation might includes memory access latency and interconncetion overhead)
        """
        total_latency = 0
        compute_power = self.wafer_scale_engine.reticle_compute_power

        for op_name, op in self._op_graph.nodes(data='operator'):
            assert op.op_type == "Matmul"
            mac_count = op.get_fp_mac_count() if forward else op.get_bp_mac_count()
            mac_count *= self.num_layer_per_pipeline_stage
            if self.zero_dp_p: mac_count *= self.data_parallel_size
            total_latency += mac_count / compute_power

        return total_latency 

    def _get_propagation_latency(self, forward: bool) -> float:
        """
        Estimation on propagation latency of single reticle running one pipeline stage
        This is a naive version where each reticle provides 100% utilization
        """
        return self._get_compute_latency(forward)
    
    def _get_weight_comm_latency(self, forward: bool) -> float:
        """
        weight communication latency of whole wafer running one pipeline stage
        """
        inter_wafer_bandwidth = self.inter_wafer_bandwidth

        weight_tensors = self._op_graph.get_tensors(kind=['weight'])
        weight_tensor_total_numel = sum([tensor.numel() for tensor in weight_tensors.values()])
        weight_tensor_total_numel *= self.tensor_parallel_size  
        if self.zero_dp_p: weight_tensor_total_numel *= self.data_parallel_size  # 1 share of complete transformer layer's weight
        weight_tensor_total_numel *= self.num_layer_per_pipeline_stage * self.num_pipeline_stage_per_wafer  # total shares on wafer

        if forward:
            if not self.zero_dp_p: return 0
            else:
                # allgather 1 share of weight with data parallel peers
                weight_size = weight_tensor_total_numel * self.training_config.get_precision_size()
                comm_volume = weight_size
                return comm_volume / inter_wafer_bandwidth
        else:
            # reduce scatter 1 share of grad and all gather 1 share of weight
            weight_size = grad_size =  weight_tensor_total_numel * self.training_config.get_precision_size()
            comm_volume = weight_size + grad_size
            return comm_volume / inter_wafer_bandwidth
        
    def _get_activation_comm_latency(self, forward: bool) -> float:
        """
        Estimation on collective communication latency of #tensor_parallel_size reticles
        This is a naive version where each link provides 100% bandwidth
        """
        T = self.tensor_parallel_size
        allreduce_bandwidth = self.wafer_scale_engine.inter_reticle_bandwidth
        tensors = self._op_graph.get_tensors()

        # consider one ring of #T tensors
        # this is already the full size
        if forward:
            allreduce_size = sum([tensor.size() for tensor in tensors.values() if tensor.name in ['Z', 'X2_mlp']])
        else:
            allreduce_size = sum([tensor.size() for tensor in tensors.values() if tensor.name in ['X', 'X0_mlp']])
        
        allreduce_size *= self.num_layer_per_pipeline_stage # 1 share of activation of a pipeline stage
        # needless to consider for zero_dp_p
        allreduce_latency = 2 * (T-1) / T * allreduce_size / allreduce_bandwidth
        return allreduce_latency
    
    def get_fp_latency(self):
        inter_wafer_bandwidth = self.inter_wafer_bandwidth
        inter_reticle_bandwidth = self.wafer_scale_engine.inter_reticle_bandwidth

        # fetch activation from previous wafer
        # All pipeline stages wait for this slowest transmission to finish
        input_tensors = self._op_graph.get_tensors(kind=['input'])
        input_tensor_total_size = sum([tensor.size() for tensor in input_tensors.values()])
        # 1 share of complete input of transformer layer
        # needless to consider for zero_dp_p
        input_inter_wafer_latency = input_tensor_total_size / inter_wafer_bandwidth

        compute_latency = self._get_propagation_latency(forward=True)
        weight_comm_latency = self._get_weight_comm_latency(forward=True)
        activation_comm_latency = self._get_activation_comm_latency(forward=True)

        # TODO: currently, swapping weight cannot be overlapped with compute
        inter_wafer_latency = input_inter_wafer_latency + weight_comm_latency
        inter_reticle_latency = activation_comm_latency
        total_latency = compute_latency + inter_wafer_latency + inter_reticle_latency

        logger.info("Summary for forward propagation latency estimation:")
        logger.info(f"- Compute latency         : {int(compute_latency * 1e9):>15d} ns ({compute_latency/total_latency:>.2%})")
        logger.info(f"- Inter-reticle comm      : {int(inter_reticle_latency * 1e9):>15d} ns ({inter_reticle_latency/total_latency:>.2%})")
        logger.info(f"- Inter-wafer comm        : {int(inter_wafer_latency * 1e9):>15d} ns ({inter_wafer_latency/total_latency:>.2%})")

        return total_latency

    def get_bp_latency(self):
        inter_wafer_bandwidth = self.inter_wafer_bandwidth
        inter_reticle_bandwidth = self.wafer_scale_engine.inter_reticle_bandwidth

        # fetch activation's grad from successive wafer
        output_tensors = self._op_graph.get_tensors(kind=['output'])
        grad_tensors_total_size = sum([tensor.size() for tensor in output_tensors.values()])
        # 1 share of complete output's grad of transformer layer
        # needless to consider for zero_dp_p
        grad_inter_wafer_latency = grad_tensors_total_size / inter_wafer_bandwidth

        # rematerialize activation
        # TODO: ZeRO-R's comm overhead is ignored, since WSE can store activation without replication
        fp_compute_latency = self._get_propagation_latency(forward=True)
        fp_weight_comm_latency = self._get_weight_comm_latency(forward=True)
        fp_activation_comm_latency = self._get_activation_comm_latency(forward=True)

        # bp latency
        bp_compute_latency = self._get_propagation_latency(forward=False)
        bp_weight_comm_latency = self._get_weight_comm_latency(forward=False)
        bp_activation_comm_latency = self._get_activation_comm_latency(forward=False)

        compute_latency = fp_compute_latency + bp_compute_latency
        inter_reticle_latency = fp_activation_comm_latency + bp_activation_comm_latency
        inter_wafer_latency = grad_inter_wafer_latency + fp_weight_comm_latency + bp_weight_comm_latency
        total_latency = compute_latency + inter_wafer_latency + inter_reticle_latency

        logger.info("Summary for backward propagation latency estimation:")
        logger.info(f"- Compute latency         : {int(compute_latency * 1e9):>15d} ns ({compute_latency/total_latency:>.2%})")
        logger.info(f"- Inter-reticle comm      : {int(inter_reticle_latency * 1e9):>15d} ns ({inter_reticle_latency/total_latency:>.2%})")
        logger.info(f"- Inter-wafer comm        : {int(inter_wafer_latency * 1e9):>15d} ns ({inter_wafer_latency/total_latency:>.2%})")

        return total_latency

    def get_dram_utilization(self) -> bool:
        # static memory
        # weight & grad & optimizer state
        weight_tensors = self._op_graph.get_tensors(kind=['weight'])
        weight_tensor_numel = sum([tensor.numel() for tensor in weight_tensors.values()])
        weight_tensor_numel *= self.tensor_parallel_size * self.num_layer_per_pipeline_stage * self.num_pipeline_stage_per_wafer
        if self.zero_dp_p: weight_tensor_numel *= self.data_parallel_size  # 1 complete share of weight

        weight_tensor_size = weight_tensor_numel * self.training_config.get_precision_size()
        if self.zero_dp_p: weight_tensor_size /= self.data_parallel_size

        weight_grad_tensor_size = weight_tensor_numel * self.training_config.get_precision_size()
        if self.zero_dp_g: weight_grad_tensor_size /= self.data_parallel_size

        weight_optimizer_state_tensor_size = weight_tensor_numel * self.training_config.get_optimizer_state_size()
        if self.zero_dp_os: weight_optimizer_state_tensor_size /= self.data_parallel_size

        # 1 share of micro batch's activation & grad
        activation_tensors = self._op_graph.get_tensors(kind=['input', 'activation', 'output'])
        activation_tensor_numel = sum([tensor.numel() for tensor in activation_tensors.values()])
        activation_tensor_numel *= self.tensor_parallel_size * self.num_layer_per_pipeline_stage * self.num_pipeline_stage_per_wafer
        activation_tensor_size = activation_tensor_numel * self.training_config.get_precision_size() * 2

        # 1 share of mini batch's input checkpoint
        checkpoint_tensors = self._op_graph.get_tensors(kind=['input'])
        checkpoint_tensor_numel = sum([tensor.numel() for tensor in checkpoint_tensors.values()])
        checkpoint_tensor_numel *= self.num_pipeline_stage_per_wafer * ceil(self.mini_batch_size / self.micro_batch_size)
        checkpoint_tensor_size = checkpoint_tensor_numel * self.training_config.get_precision_size()

        utilized_memory = weight_tensor_size + weight_grad_tensor_size + weight_optimizer_state_tensor_size + activation_tensor_size + checkpoint_tensor_size
        total_memory = self.wafer_scale_engine.dram_size

        logger.info("Summary for checking DRAM utilization:")
        logger.info(f"weight                  : {int(weight_tensor_size / 1e9):>15d} GB ({weight_tensor_size / total_memory:.2%})")
        logger.info(f"weight's grad           : {int(weight_grad_tensor_size / 1e9):>15d} GB ({weight_grad_tensor_size/ total_memory:.2%})")
        logger.info(f"weight's optimizer state: {int(weight_optimizer_state_tensor_size / 1e9):>15d} GB ({weight_optimizer_state_tensor_size/ total_memory:.2%})")
        logger.info(f"activation & grad       : {int(activation_tensor_size / 1e9):>15d} GB ({activation_tensor_size / total_memory:.2%})")
        logger.info(f"checkpoint              : {int(checkpoint_tensor_size / 1e9):>15d} GB ({checkpoint_tensor_size / total_memory:.2%})")
        logger.info(f"Total utilized DRAM     : {int(utilized_memory / 1e9):>15d} GB ({utilized_memory / total_memory:.2%})")

        return utilized_memory <= total_memory
    
    def get_training_throughput(self) -> float:
        """ Use GPipe's pipelining technique.
        """
        logger.info("Calculating training throughput of attention module")

        single_stage_fp_latency = self.get_fp_latency()
        single_stage_bp_latency = self.get_bp_latency()
        pipeline_factor = (self.model_parallel_size - 1) + ceil(self.mini_batch_size / self.micro_batch_size)
        whole_batch_latency = pipeline_factor * (single_stage_fp_latency + single_stage_bp_latency)
        sequence_per_sec = self.mini_batch_size / whole_batch_latency

        return sequence_per_sec

    def get_training_wse_utilization(self) -> float:
        logger.info("Calculating training wse utilization of attention module")
        
        single_stage_fp_latency = self.get_fp_latency()
        single_stage_bp_latency = self.get_bp_latency()
        pipeline_factor = (self.model_parallel_size - 1) + ceil(self.mini_batch_size / self.micro_batch_size)
        whole_batch_latency = pipeline_factor * (single_stage_fp_latency + single_stage_bp_latency)

        compute_latency = self._get_compute_latency(forward=True) * 2 + self._get_compute_latency(forward=False)
        compute_latency *= ceil(self.mini_batch_size / self.micro_batch_size)
        utilization = compute_latency / whole_batch_latency

        return utilization
    
class ReticleFidelityWseTransformerRunner(WseTransformerRunner):
    def __init__(self, 
                 attention_heads: int,
                 hidden_size: int, 
                 sequence_length: int, 
                 number_of_layers: int, 
                 micro_batch_size: int, 
                 mini_batch_size: int, 
                 data_parallel_size: int, 
                 model_parallel_size: int, 
                 tensor_parallel_size: int, 
                 wafer_scale_engine: WaferScaleEngine, 
                 inter_wafer_bandwidth: int, 
                 training_config: TrainingConfig, 
                 zero_dp_os: bool = True, 
                 zero_dp_g: bool = True, 
                 zero_dp_p: bool = False, 
                 zero_r_pa: bool = True
                 ) -> None:
        super().__init__(attention_heads, hidden_size, sequence_length, number_of_layers, micro_batch_size, mini_batch_size, data_parallel_size, model_parallel_size, tensor_parallel_size, wafer_scale_engine, inter_wafer_bandwidth, training_config, zero_dp_os, zero_dp_g, zero_dp_p, zero_r_pa)
        assert self.zero_dp_p == False, "Each wafer must hold a full copy of weight"
        assert tensor_parallel_size <= wafer_scale_engine.reticle_array_height * wafer_scale_engine.reticle_array_width, "Cannot tensor parallel on a tiny wafer!"
        self.virtual_reticle_id_2_parallel_index = {i: (t, m) for i, (t, m) in enumerate(product(range(tensor_parallel_size), range(self.num_pipeline_stage_per_wafer)))}
        self.parallel_index_2_virtual_reticle_id = {(t, m): i for i, (t, m) in enumerate(product(range(tensor_parallel_size), range(self.num_pipeline_stage_per_wafer)))}

    def __create_propagation_reticle_task(self, virtual_reticle_id: int, forward: bool) -> FusedReticleTask:
        tensor_parallel_id, model_parallel_id = self.virtual_reticle_id_2_parallel_index[virtual_reticle_id]

        task_graph = nx.DiGraph()  # for simplicity, there's no edge
        def add_subtask(name: str, subtask: BaseReticleTask):
            task_graph.add_node(name, task=subtask)

        # read input from DRAM or previous reticle
        # needless to assign read & write to the same link
        if forward:
            read_input_data_amount = sum([tensor.numel() * self.training_config.get_precision_size() 
                                for tensor in self._op_graph.get_tensors(kind=['input']).values()])
            read_input_data_amount //= self.micro_batch_size
            if model_parallel_id == 0:
                # We assume each virtual reticle uses its private virtual dram port
                read_input_subtask = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port=virtual_reticle_id, 
                                                        access_type='read', data_amount=read_input_data_amount)
            else:
                peer_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id((tensor_parallel_id, model_parallel_id - 1))
                read_input_subtask = PeerAccessReticleTask(virtual_reticle_id, peer_virtual_reticle_id, 'read', read_input_data_amount)
            add_subtask('read_input', read_input_subtask)
        else:
            read_outgrad_data_amount = sum([tensor.numel() * self.training_config.get_precision_size() 
                                for tensor in self._op_graph.get_tensors(kind=['output']).values()])
            read_outgrad_data_amount //= self.micro_batch_size
            if model_parallel_id == self.num_pipeline_stage_per_wafer - 1:
                read_outgrad_subtask = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port=virtual_reticle_id, 
                                                        access_type='read', data_amount=read_outgrad_data_amount)
            else:
                peer_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id((tensor_parallel_id, model_parallel_id + 1))
                read_outgrad_subtask = PeerAccessReticleTask(virtual_reticle_id, peer_virtual_reticle_id, 'read', read_outgrad_data_amount)
            add_subtask('read_outgrad', read_outgrad_subtask)

        # compute
        # allocate cores according to compute amount
        # The time of running multiple compute tasks on different amount of cores is determined by the slowest one.
        # This time can also be calculated with a refractored compute amount and the reticle compute power.
        attn_mac_count = sum([op.get_fp_mac_count() if forward else op.get_bp_mac_count() 
                              for name, op in self._op_graph.nodes(data='operator') 
                              if name in ['linear_qkv', 'matmul_qk', 'matmul_scorev', 'linear_proj']]) // self.micro_batch_size
        mlp_mac_count = sum([op.get_fp_mac_count() if forward else op.get_bp_mac_count() 
                             for name, op in self._op_graph.nodes(data='operator') 
                             if name in ['linear_mlp0', 'linear_mlp1']]) // self.micro_batch_size
        attn_mac_ratio = 1 / self.num_layer_per_pipeline_stage * (attn_mac_count / (attn_mac_count + mlp_mac_count))
        equivalent_compute_amount = attn_mac_count / attn_mac_ratio
        compute_task = ComputeReticleTask(virtual_reticle_id, equivalent_compute_amount)
        add_subtask('compute', compute_task)

        # swap weight if necessary
        activation_size = sum([tensor.numel() * self.training_config.get_precision_size()
                               for tensor in self._op_graph.get_tensors(kind=['input', 'activation', 'output']).values()])
        activation_size = activation_size // self.micro_batch_size * self.num_layer_per_pipeline_stage
        weight_size = sum([tensor.numel() * self.training_config.get_precision_size()
                           for tensor in self._op_graph.get_tensors(kind=['weight']).values()])
        weight_size = weight_size * self.num_layer_per_pipeline_stage
        reticle_sram_size = Reticle.get_sram_size(self.wafer_scale_engine.reticle_config)
        assert activation_size <= reticle_sram_size, f"Reticle sram {reticle_sram_size} cannot hold even one sequence's activation {activation_size}"
        if forward:
            if activation_size + weight_size > reticle_sram_size:
                read_weight_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port=virtual_reticle_id, 
                                                        access_type='read', data_amount=weight_size)
                add_subtask('read_weight', read_weight_task)
        else:
            if activation_size + 2 * weight_size > reticle_sram_size:
                read_weight_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port=virtual_reticle_id, 
                                                        access_type='read', data_amount=weight_size)
                write_weightgrad_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port=virtual_reticle_id, 
                                                        access_type='write', data_amount=weight_size)
                add_subtask('read_weight', read_weight_task)
                add_subtask('write_weightgrad', write_weightgrad_task)
        
        # allreduce output
        allreduce_size = sum([tensor.numel() * self.training_config.get_precision_size() 
                              for tensor in self._op_graph.get_tensors().values() if tensor.name in ['Z', 'X2_mlp']])
        allreduce_size //= self.micro_batch_size
        allreduce_size *= self.num_layer_per_pipeline_stage
        next_peer_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id[((tensor_parallel_id + 1) % self.tensor_parallel_size, model_parallel_id)]
        allreduce_task = PeerAccessReticleTask(virtual_reticle_id, next_peer_virtual_reticle_id, 'write', allreduce_size)
        add_subtask('allreduce', allreduce_task)

        fused_task = FusedReticleTask(virtual_reticle_id, task_graph, repeated_times=self.micro_batch_size)
        return fused_task

    def __create_weight_update_reticle_task(self, virtual_reticle_id: int) -> FusedReticleTask:
        task_graph = nx.DiGraph()  # for simplicity, there's no edge
        def add_subtask(name: str, subtask: BaseReticleTask):
            task_graph.add_node(name, task=subtask)
            
        precision = self.training_config.get_precision_size()
        optimizer_state_factor = self.training_config.get_optimizer_state_size()
        compute_factor = self.training_config.get_weight_update_compute_amount()
    
        weight_numel = sum([tensor.numel() for tensor in self._op_graph.get_tensors(kind=['weight']).values()])
        weight_numel *= self.num_layer_per_pipeline_stage
        dram_access_amount = weight_numel * (precision + optimizer_state_factor)
        compute_amount = weight_numel * compute_factor

        read_subtask = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port=virtual_reticle_id, 
                                             access_type='read', data_amount=dram_access_amount)
        write_subtask = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port=virtual_reticle_id, 
                                             access_type='write', data_amount=dram_access_amount)
        compute_subtask = ComputeReticleTask(virtual_reticle_id, compute_amount)
        add_subtask('read', read_subtask)
        add_subtask('write', write_subtask)
        add_subtask('compute', compute_subtask)
        fused_task = FusedReticleTask(virtual_reticle_id, task_graph, repeated_times=self.micro_batch_size)
        return fused_task


    def __create_three_stage_task_generator_from_matmul_operator(self, op: MatMulOperator, forward: bool):
        # Deprecated 
        precision = self.training_config.get_precision_size()
        input_tensor_size = sum([tensor.numel() * precision for tensor in op.input_tensors.values()])
        output_tensor_size = sum([tensor.numel() * precision for tensor in op.output_tensors.values()])
        if forward:
            compute_amount = op.get_fp_mac_count()
            read_data_amount = [input_tensor_size]
            write_data_amount = [output_tensor_size]
        else:
            compute_amount = op.get_bp_mac_count()
            read_data_amount = [input_tensor_size + output_tensor_size]  # input & output's grad
            write_data_amount = [input_tensor_size]  # input's grad
        kwargs = {
            'compute_amount': compute_amount,
            'read_data_amount': read_data_amount,
            'write_data_amount': write_data_amount,
            'reuse_dram_port': False,
        }
        return ThreeStageReticleTaskGenerator(**kwargs)
    
    def __create_three_stage_task_generator_from_weight_update(self, weight: TensorInfo):
        precision = self.training_config.get_precision_size()
        optimizer_state_factor = self.training_config.get_optimizer_state_size()
        compute_factor = self.training_config.get_weight_update_compute_amount()
        dram_access_amount = [weight.numel() * (precision + optimizer_state_factor)]
        compute_amount = weight.numel() * compute_factor
        kwargs = {
            'compute_amount': compute_amount,
            'read_data_amount': dram_access_amount,
            'write_data_amount': dram_access_amount,
            'reuse_dram_port': False,
        }
        return ThreeStageReticleTaskGenerator(**kwargs)

    def _get_propagation_latency(self, forward: bool) -> float:
        """
        Estimation on propagation latency of single reticle running one pipeline stage
        Consider all reticles' interaction on the wafer.
        """
        total_latency = 0

        # propagation latency 
        wse_task = ListWaferTask([self.__create_propagation_reticle_task(vrid, forward=True) for vrid in self.virtual_reticle_id_2_parallel_index])
        mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
        wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
        total_latency += wse_evaluator.get_total_latency()
        
        if not forward:
            # consider updating weights
            wse_task = ListWaferTask([self.__create_weight_update_reticle_task(vrid) for vrid in self.virtual_reticle_id_2_parallel_index])
            mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
            wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
            total_latency += wse_evaluator.get_total_latency()

        return total_latency
    
    def _get_activation_comm_latency(self, forward: bool) -> float:
        return 0  # overlapped with computation