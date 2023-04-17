
from math import ceil

from typing import Dict, Tuple, Union, List
from itertools import product
from functools import reduce
import networkx as nx
from copy import deepcopy

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
                 training_config: TrainingConfig,
                 inter_wafer_bandwidth: Union[int, None] = None,
                 zero_dp_os: bool = True,
                 zero_dp_g: bool = True,
                 zero_dp_p: bool = False,
                 zero_r_pa: bool = True,
                 num_reticle_per_model_chunk: int = 1,
                 weight_streaming: bool = True,
                 **kwargs,
                 ) -> None:
        # model parameters
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.number_of_layers = number_of_layers
        assert hidden_size % attention_heads == 0

        # parallelism parameters
        self.mini_batch_size = mini_batch_size    # unit of updating weight
        self.micro_batch_size = micro_batch_size  # unit of model pipelining
        assert micro_batch_size <= mini_batch_size

        self.data_parallel_size = data_parallel_size      # inter-wafer
        self.model_parallel_size = model_parallel_size    # inter-wafer
        self.tensor_parallel_size = tensor_parallel_size  # intra-wafer
        self.num_reticle_per_model_chunk = num_reticle_per_model_chunk
        # check valid model chunking
        assert number_of_layers % model_parallel_size == 0
        assert hidden_size % tensor_parallel_size == 0
        # check available reticle
        used_reticle = self.tensor_parallel_size * self.num_reticle_per_model_chunk
        total_reticle = self.wafer_scale_engine.reticle_array_height * self.wafer_scale_engine.reticle_array_width
        assert used_reticle <= total_reticle, f"Requiring {used_reticle} reticles but there's only {total_reticle}"
        self.num_layer_per_model_chunk = (self.number_of_layers // self.model_parallel_size)

        # intra-model-chunk params, transparent to users
        self.nano_batch_size = None
        self.layer_pipeline_size = None

        self.weight_streaming = weight_streaming

        # ZeRO settings, this setting is now fixed for simplicity
        self.zero_dp_os = zero_dp_os
        self.zero_dp_g = zero_dp_g
        self.zero_dp_p = zero_dp_p
        self.zero_r_pa = zero_r_pa
        assert zero_dp_os is True
        assert zero_dp_g is True
        assert zero_dp_p is False
        assert zero_r_pa is True

        # wse platform
        self.wafer_scale_engine = wafer_scale_engine
        self.inter_wafer_bandwidth = inter_wafer_bandwidth if inter_wafer_bandwidth \
                                     else (wafer_scale_engine.reticle_array_height * wafer_scale_engine.reticle_array_width * wafer_scale_engine.inter_reticle_bandwidth)  # assume wafer-array bandwidth

        # training settings 
        self.training_config = training_config
    
    # TODO: find optimal nano batch size and layer pipeline stage, and remove these API
    def _set_nano_batch_size(self, nano_batch_size) -> None:
        """ reset nano batch size and check feasibility
        """
        assert nano_batch_size <= self.micro_batch_size
        self.nano_batch_size = nano_batch_size

    def _set_layer_pipeline_size(self, layer_pipeline_size) -> None:
        """ reset layer pipeline size and relevant params, and check feasibility
        """
        # check division feasibility
        assert (self.number_of_layers // self.model_parallel_size) % layer_pipeline_size == 0
        self.layer_pipeline_size = layer_pipeline_size

    # helper functions for API

    def _get_flops_per_seq_per_layer(self, inference=False):
        """ Calculate FLOPS, copied from Cerebras-GPT appendix.
        Embedding and final logits are ignored for simplicity.
        """
        d_model = self.hidden_size
        seq_len = self.sequence_length
        num_heads = self.attention_heads
        key_size = d_model // num_heads

        kqv_proj = 2 * 3 * seq_len * d_model * (key_size * num_heads)  # 2 for mul+add
        kq_logits = 2 * (seq_len ** 2) * (key_size * num_heads)
        softmax = 3 * (key_size * num_heads) * (seq_len ** 2)  # 3-passes
        softmax_q_red = (seq_len ** 2) * (key_size * num_heads)
        final_linear = 2 * seq_len * (key_size * num_heads) * d_model
        sm_v_dot = 2 * (seq_len ** 2) * (key_size * num_heads)  # dot mul ! I've made a mistake before
        attention_blocks = kqv_proj + kq_logits + softmax + softmax_q_red + final_linear + sm_v_dot

        dense_blocks = 16 * seq_len * (d_model ** 2)

        # Layer norms: 7 FLOPs/activation, 2 LNs per decoder block 
        layer_norm_flops = 2 * 7 * (seq_len * d_model) 
        # GeLU: estimate 20 FLOPs/activation, applied in FFN with 4x hidden dim 
        gelu_flops = 20 * 4 * (seq_len * d_model)

        total_flops = attention_blocks + dense_blocks + layer_norm_flops + gelu_flops

        if inference:
            return total_flops
        else:
            return total_flops * 3  # fp + bp + rematerialization
    
    def _get_ideal_compute_latency(self, inference: bool) -> float:
        """ Ideal compute latency of a model chunk
        Assume 100% compute resource utilization.
        """
        assert self.zero_dp_p is False

        compute_power = self.wafer_scale_engine.reticle_compute_power * self.num_reticle_per_model_chunk
        total_flops = self._get_flops_per_seq_per_layer(inference) * self.micro_batch_size * self.num_layer_per_model_chunk
        total_latency = total_flops / compute_power

        return total_latency 

    def _get_weight_numel_per_model_chunk(self):
        d_model = self.hidden_size

        ln1 = 2 * d_model 
        attn = 4 * (d_model ** 2 + d_model) // self.tensor_parallel_size
        ln2 = 2 * d_model 
        ffn = 8 * (d_model ** 2) + 5 * d_model // self.tensor_parallel_size
        
        total_size = ln1 + attn + ln2 + ffn
        total_size *= self.num_layer_per_model_chunk

        return total_size
    
    def _get_attn_activation_numel_per_model_chunk_per_seq_per_layer(self) -> int:
        d_model = self.hidden_size
        seq_len = self.sequence_length
        num_heads = self.attention_heads // self.tensor_parallel_size
        key_size = d_model // self.attention_heads

        x = seq_len * d_model
        kqv = 3 * seq_len * (key_size * num_heads)
        kq_logits = softmax = (seq_len ** 2) * (key_size * num_heads)
        sm_v_dot = seq_len * (key_size * num_heads)
        final_linear = seq_len * d_model

        total = x + kqv + kq_logits + softmax + sm_v_dot + final_linear
        return total

    def _get_ffn_activation_numel_per_model_chunk_per_seq_per_layer(self) -> int:
        d_model = self.hidden_size
        seq_len = self.sequence_length

        x = seq_len * d_model
        x_ = y = seq_len * 4 * d_model // self.tensor_parallel_size  # GELU
        z = seq_len * d_model

        total = x + x_ + y + z
        return total

    # Calculate latency relevant to each model chunk

    def _get_input_latency(self, inference: bool) -> float:
        """ This latency is dominated by fetching 1 share of input, 
            and broadcasting to other tensor parallel model chunk.
        """
        input_tensor_numel = self.micro_batch_size * self.sequence_length * self.hidden_size
        if not inference: input_tensor_numel *= 2  # rematerialization

        dram_access_size = input_tensor_numel * self.training_config.get_precision_size()
        dram_access_latency = dram_access_size / self.wafer_scale_engine.dram_bandwidth

        inter_reticle_size = input_tensor_numel * self.training_config.get_precision_size() * (self.tensor_parallel_size - 1)
        bisection_bandwidth = self.wafer_scale_engine.get_bisection_bandwidth()
        inter_reticle_latency = inter_reticle_size / bisection_bandwidth

        total_latency = dram_access_latency + inter_reticle_latency
        return total_latency
    
    def _get_swap_weight_latency(self, inference: bool) -> float:
        """
        Estimation on swapping weight in and out of DRAM.
        """
        assert self.weight_streaming
        assert self.nano_batch_size, "You should set nano batch size first"

        weight_numel_per_model_chunk = self._get_weight_numel_per_model_chunk()
        weight_numel_per_wafer = weight_numel_per_model_chunk * self.tensor_parallel_size
        swapped_weight_numel = ceil(self.micro_batch_size / self.nano_batch_size) * weight_numel_per_wafer
        if not inference: swapped_weight_numel *= 3  # fp-read + rebuild + bp-read + bp write grad, we fuse rebuild and bp-read for simplicity
        swapped_weight_size = swapped_weight_numel * self.training_config.get_precision_size()

        bisection_bandwidth = self.wafer_scale_engine.get_bisection_bandwidth()
        total_dram_bandwidth = self.wafer_scale_engine.get_total_dram_bandwidth()
        bottleneck_bandwidth = min(bisection_bandwidth, total_dram_bandwidth)

        total_latency = swapped_weight_size / bottleneck_bandwidth

        return total_latency
    
    def _get_compute_latency(self, inference: bool) -> float:
        return self._get_ideal_compute_latency(inference)
    
    def _get_activation_allreduce_latency(self, inference: bool) -> float:
        """
        Estimation on collective communication latency of #tensor_parallel_size reticles
        This is a naive version where each link provides 100% bandwidth
        """
        T = self.tensor_parallel_size
        allreduce_bandwidth = self.wafer_scale_engine.inter_reticle_bandwidth * self.num_reticle_per_model_chunk * T  # ring-allreduce

        allreduce_numel_per_model_chunk = self.micro_batch_size * self.sequence_length * self.hidden_size * 2 * self.num_layer_per_model_chunk  # attn, ffn
        allreduce_numel = allreduce_numel_per_model_chunk * 2 * (T - 1)
        if not inference: allreduce_numel *= 2
        allreduce_size = allreduce_numel * self.training_config.get_precision_size()

        allreduce_latency = allreduce_size / allreduce_bandwidth
        
        return allreduce_latency
    
    # inter-wafer communication latency

    def _get_weight_update_latency(self) -> float:
        """ This is completely transmission-bottlenecked, so we only consider allreduce latency between data parallel workers
        """
        D = self.data_parallel_size
        ring_collective_bandwidth = self.inter_wafer_bandwidth * D

        weight_numel_per_model_chunk = self._get_weight_numel_per_model_chunk()
        weight_numel_per_wafer = weight_numel_per_model_chunk * self.tensor_parallel_size

        # reduce scatter 1 share of grad and all gather 1 share of weight
        weight_size = grad_size =  weight_numel_per_wafer * self.training_config.get_precision_size()
        ring_collective_size = (D - 1) * (weight_size + grad_size)
    
        total_latency = ring_collective_size / ring_collective_bandwidth
        return total_latency

    def _get_activation_cross_pipeline_stage_latency(self):
        comm_size = self.micro_batch_size * self.sequence_length * self.hidden_size  # same for fp & bp

        bandwidth = self.inter_wafer_bandwidth

        comm_latency = comm_size / bandwidth
        return comm_latency
    
    def get_sram_utilization(self, inference: bool, nano_batch_size: int = None, layer_pipeline_size: int = None) -> bool:
        """ Check if SRAM could hold a model chunk
        """
        sram_size = Reticle.get_sram_size(self.wafer_scale_engine.reticle_config) * self.num_reticle_per_model_chunk
        precision = self.training_config.get_precision_size()

        if nano_batch_size is None: nano_batch_size = self.nano_batch_size
        if layer_pipeline_size is None: layer_pipeline_size = self.layer_pipeline_size
        assert nano_batch_size
        assert layer_pipeline_size

        attn_act_size = self._get_attn_activation_numel_per_model_chunk_per_seq_per_layer() * precision
        ffn_act_size = self._get_ffn_activation_numel_per_model_chunk_per_seq_per_layer() * precision
        if self.weight_streaming:
            assert layer_pipeline_size == 1, "You shouldn't set layer pipelining on weight streaming mode!"
            if inference:
                # only store 1 layer's activation
                act_size = max(attn_act_size, ffn_act_size) * nano_batch_size
            else:
                # buffer all activations
                act_size = (attn_act_size + ffn_act_size) * self.num_layer_per_model_chunk * self.nano_batch_size
            return act_size < sram_size  # save some space for weight
        else:  # layer pipelining
            weight_size = self._get_weight_numel_per_model_chunk()
            # each pipeline stage hold one complete share of activation
            act_size = layer_pipeline_size * nano_batch_size * (attn_act_size + layer_pipeline_size)
            if inference:
                return weight_size + act_size <= sram_size
            else:
                # each layer pipeline stage need to buffer the same number of nano batch as the depth of pipeline
                checkpoint_size = layer_pipeline_size * (layer_pipeline_size * nano_batch_size ) * self.sequence_length * self.hidden_size
                return checkpoint_size + weight_size + act_size <= sram_size
    
    def get_propagation_latency(self, forward=True, detailed_report=False):
        """ Only consider events WITHIN a reticle.
        Naive version doesn't consider overlapping, but lp_solver version does
        """
        need_to_swap_weight, need_to_swap_activation = self.get_sram_utilization(forward)
        input_latency = self._get_input_latency(forward=forward)
        swap_weight_latency = self._get_swap_weight_latency(forward=forward) if need_to_swap_weight else 0 
        swap_activation_latency = self._get_swap_activation_latency(forward=forward) if need_to_swap_activation else 0 
        compute_latency = self._get_compute_latency(forward=forward)
        activation_allreduce_latency = self._get_activation_allreduce_latency(forward=forward)

        total_latency = compute_latency + activation_allreduce_latency + swap_weight_latency + swap_activation_latency + input_latency
        if detailed_report:
            return {
                'compute': compute_latency,
                'inter_reticle': input_latency + activation_allreduce_latency + swap_weight_latency + swap_activation_latency,
            }
        else:
            return total_latency
    
    # API for evaluations

    def get_dram_utilization(self) -> bool:
        # static memory
        # weight & grad & optimizer state
        assert self.zero_dp_p is False

        # weight / grad / optimizer state on the wafer
        weight_tensors = self._op_graph.get_tensors(kind=['weight'])
        weight_tensor_numel = sum([tensor.numel() for tensor in weight_tensors.values()])
        weight_tensor_numel *= self.tensor_parallel_size * self.num_layer_per_pipeline_stage * self.num_pipeline_stage_per_wafer

        weight_tensor_size = weight_tensor_numel * self.training_config.get_precision_size()

        weight_grad_tensor_size = weight_tensor_numel * self.training_config.get_precision_size()
        if self.zero_dp_g: weight_grad_tensor_size /= self.data_parallel_size

        weight_optimizer_state_tensor_size = weight_tensor_numel * self.training_config.get_optimizer_state_size()
        if self.zero_dp_os: weight_optimizer_state_tensor_size /= self.data_parallel_size

        # 1 share of micro batch's activation & grad
        # debug: activation should be finely selected in case of repeated counting
        activation_tensors = self._op_graph.get_tensors(kind=['input', 'activation', 'output'])
        activation_tensor_numel = sum([tensor.numel() for tensor in activation_tensors.values() if tensor.name in ['X', 'QKV', 'scores', 'Y', 'Z', 'X0_mlp', 'X1_mlp', 'X2_mlp']])
        activation_tensor_numel *= self.tensor_parallel_size * self.num_layer_per_pipeline_stage * self.num_pipeline_stage_per_wafer
        activation_tensor_size = activation_tensor_numel * self.training_config.get_precision_size() * 2
        # logger.warning("Activation tensors will no longer be stored in DRAM in future version.")
        activation_tensor_size = 0

        # 1 share of mini batch's input checkpoint
        checkpoint_tensors = self._op_graph.get_tensors(kind=['input'])
        checkpoint_tensor_numel = sum([tensor.numel() for tensor in checkpoint_tensors.values()])
        checkpoint_tensor_numel *= self.num_pipeline_stage_per_wafer * (self.mini_batch_size / self.micro_batch_size)
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

        single_stage_fp_latency = self.get_propagation_latency(forward=True) + self._get_activation_cross_pipeline_stage_latency(connect_type='wafer')
        single_stage_bp_latency = self.get_propagation_latency(forward=False) + self._get_activation_cross_pipeline_stage_latency(connect_type='wafer')
        weight_comm_latency = self._get_weight_update_latency() 
        pipeline_factor = (self.model_parallel_size - 1) + ceil(self.mini_batch_size / self.micro_batch_size / self.data_parallel_size)
        whole_batch_latency = pipeline_factor * (single_stage_fp_latency + single_stage_bp_latency) + weight_comm_latency
        sequence_per_sec = (self.mini_batch_size / self.data_parallel_size) / whole_batch_latency

        return sequence_per_sec

    def get_training_wse_utilization(self) -> float:
        logger.info("Calculating training wse utilization of attention module")
        
        single_stage_fp_latency = self.get_propagation_latency(forward=True) + self._get_activation_cross_pipeline_stage_latency(connect_type='wafer')
        single_stage_bp_latency = self.get_propagation_latency(forward=False) + self._get_activation_cross_pipeline_stage_latency(connect_type='wafer')
        weight_update_latency = self._get_weight_update_latency() 
        pipeline_factor = (self.model_parallel_size - 1) + ceil(self.mini_batch_size / self.micro_batch_size / self.data_parallel_size)
        whole_batch_latency = pipeline_factor * (single_stage_fp_latency + single_stage_bp_latency) + weight_update_latency

        ideal_compute_latency = self._get_ideal_compute_latency(forward=True) * 2 + self._get_ideal_compute_latency(forward=False)
        ideal_compute_latency *= ceil(self.mini_batch_size / self.micro_batch_size)
        utilization = ideal_compute_latency / whole_batch_latency

        # debugging
        fp_report = self.get_propagation_latency(forward=True, detailed_report=True)
        bp_report = self.get_propagation_latency(forward=False, detailed_report=True)
        full_report = {
            'compute': ceil(self.mini_batch_size / self.micro_batch_size) * (fp_report['compute'] + bp_report['compute']),
            'inter_reticle': ceil(self.mini_batch_size / self.micro_batch_size) * (fp_report['inter_reticle'] + bp_report['inter_reticle']),
            'inter_wafer': ceil(self.mini_batch_size / self.micro_batch_size) * (self._get_activation_cross_pipeline_stage_latency(connect_type='wafer') * 2) + weight_update_latency,
            'idle': (self.model_parallel_size - 1) * (single_stage_fp_latency + single_stage_bp_latency),
        }
        logger.info("Summary for checking time utilization proportion:")
        for k, v in full_report.items():
            logger.info(f"{k:<20} : {int(v*1e3):>15d} ms ({v/whole_batch_latency:.2%})")

        return utilization
    
    def get_inference_latency(self) -> float:
        logger.info("Calculating inference latency of attention module")

        single_stage_fp_latency = self.get_propagation_latency(forward=True)
        total_fp_latency = single_stage_fp_latency * self.model_parallel_size
        total_cross_wafer_times = ceil(self.model_parallel_size / self.num_pipeline_stage_per_wafer) + 1  # in, cross_wafer, out
        total_cross_reticle_times = self.model_parallel_size - (total_cross_wafer_times - 1)
        cross_wafer_latency = self._get_activation_cross_pipeline_stage_latency(connect_type='wafer') * total_cross_wafer_times
        cross_reticle_latency = self._get_activation_cross_pipeline_stage_latency(connect_type='reticle') * total_cross_reticle_times

        total_latency = total_fp_latency + cross_wafer_latency + cross_reticle_latency
        return total_latency

    # TODO: power API
    
class ReticleFidelityWseTransformerRunner(WseTransformerRunner):
    """ Estimate WSE performance with higher fidelity
    Propagation latency now considers more inter-reticle transmission features: 
    - Traffic induced by accessing DRAM
    - Congestion in allreduce induced by inappropiate mapping
    - Overlapping computation and inter-reticle communication

    TODO: All I need to write is get_propagation_latency
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
                 training_config: TrainingConfig, 
                 inter_wafer_bandwidth: Union[int, None] = None, 
                 zero_dp_os: bool = True, 
                 zero_dp_g: bool = True, 
                 zero_dp_p: bool = False, 
                 zero_r_pa: bool = True, 
                 num_reticle_per_model_chunk: int = 1,
                 **kwargs,
                 ) -> None:
        super().__init__(attention_heads, hidden_size, sequence_length, number_of_layers, micro_batch_size, mini_batch_size, data_parallel_size, model_parallel_size, tensor_parallel_size, wafer_scale_engine, training_config, inter_wafer_bandwidth, zero_dp_os, zero_dp_g, zero_dp_p, zero_r_pa, num_reticle_per_model_chunk)
        self.virtual_reticle_id_2_parallel_index = {i: (m, t, r) for i, (m, t, r) in enumerate(product(range(self.num_pipeline_stage_per_wafer), range(tensor_parallel_size), range(self.num_reticle_per_model_chunk)))}
        self.parallel_index_2_virtual_reticle_id = {(m, t, r): i for i, (m, t, r) in enumerate(product(range(self.num_pipeline_stage_per_wafer), range(tensor_parallel_size), range(self.num_reticle_per_model_chunk)))}
        self.is_overlap = True  # if ture, we overlap compute with inter reticle communication
        self.__virtual_dram_port_counter = 0
    
    def __alloc_new_dram_port(self):
        vdp = deepcopy(self.__virtual_dram_port_counter)
        self.__virtual_dram_port_counter += 1
        return vdp
    
    def __assign_input_reticle_task(self, forward: bool) -> List[BaseReticleTask]:
        """ Reticle tasks for input
        """

        # activation size of 1 sequence input
        tensors = self._op_graph.get_tensors()
        input_tensor_numel = sum([tensor.numel() for tensor in tensors.values() if tensor.name in ['X']])
        input_tensor_numel = input_tensor_numel / self.micro_batch_size
        input_tensor_size = input_tensor_numel * self.training_config.get_precision_size()

        def need_to_access_dram(worker_reticle_index) -> bool:
            if forward:
                return worker_reticle_index == 0
            else:
                return worker_reticle_index == self.num_reticle_per_model_chunk - 1

        task_list = []

        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            pipeline_stage_index, tensor_parallel_index, worker_reticle_index = parallel_index
            if need_to_access_dram(worker_reticle_index=worker_reticle_index):  # need to fetch input from DRAM
                if tensor_parallel_index == 0:  # only the first worker access DRAM
                    virtual_dram_port = self.__alloc_new_dram_port()
                    reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'read', input_tensor_size, repeated_times=self.micro_batch_size)
                    task_list.append(reticle_task)
                else:  # other workers fetch from their peers, forming a binary tree
                    pred_tensor_parallel_index = tensor_parallel_index - 1  # but that would make transmission overlap, and we simply assumes 100% bandwidth utilization
                    pred_parallel_index = (pipeline_stage_index, pred_tensor_parallel_index, worker_reticle_index)
                    pred_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id[pred_parallel_index]
                    reticle_task = PeerAccessReticleTask(virtual_reticle_id, pred_virtual_reticle_id, 'read', input_tensor_size, repeated_times=self.micro_batch_size)
                    task_list.append(reticle_task)
            else:  # other workers uses predcessor's activation
                pred_shift = -1 if forward else 1
                pred_worker_reticle_index = worker_reticle_index + pred_shift
                pred_parallel_index = (pipeline_stage_index, tensor_parallel_index, pred_worker_reticle_index)
                pred_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id[pred_parallel_index]
                reticle_task = PeerAccessReticleTask(virtual_reticle_id, pred_virtual_reticle_id, 'read', input_tensor_size, repeated_times=self.micro_batch_size)
                task_list.append(reticle_task)

        # rematerialization input tasks
        if forward is False:
            task_list += self.__assign_input_reticle_task(forward=True)

        return task_list
    
    def __assign_swap_weight_reticle_task(self, forward: bool) -> List[BaseReticleTask]:
        # weight size for one reticle
        # this split is very naive, and may not lead to a valid split
        tensors = self._op_graph.get_tensors()
        weight_tensor_numel = sum([tensor.numel() for tensor in tensors.values() if tensor.name in ['W_qkv', 'W_proj', 'W0_mlp', 'W1_mlp']])
        weight_tensor_numel = weight_tensor_numel * self.num_layer_per_pipeline_stage / self.num_reticle_per_model_chunk
        weight_tensor_size = weight_tensor_numel * self.training_config.get_precision_size()
        
        task_list = []

        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            virtual_dram_port = self.__alloc_new_dram_port()
            fp_reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'read', weight_tensor_size, repeated_times=self.micro_batch_size)
            task_list.append(fp_reticle_task)
            if not forward:
                # write grad back to DRAM
                bp_reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'write', weight_tensor_size, repeated_times=self.micro_batch_size)
                task_list.append(bp_reticle_task)

        return task_list
    
    def __assign_swap_activation_reticle_task(self, forward: bool) -> List[BaseReticleTask]:
        tensors = self._op_graph.get_tensors()
        activation_tensor_numel = sum([tensor.numel() for tensor in tensors.values() if tensor.name in ['X', 'QKV', 'scores', 'Y', 'Z', 'X0_mlp', 'X1_mlp', 'X2_mlp']])
        activation_tensor_numel = activation_tensor_numel * self.num_layer_per_pipeline_stage / self.num_reticle_per_model_chunk / self.micro_batch_size
        activation_tensor_size = activation_tensor_numel * self.training_config.get_precision_size()
        activation_tensor_size = activation_tensor_size * 2 if not forward else activation_tensor_size 

        task_list = []

        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            virtual_dram_port = self.__alloc_new_dram_port()
            write_reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'write', activation_tensor_size, repeated_times=self.micro_batch_size)
            read_reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'read', activation_tensor_size, repeated_times=self.micro_batch_size)
            task_list.append(read_reticle_task)
            task_list.append(write_reticle_task)

        return task_list

    def __assign_compute_reticle_task(self, forward: bool) -> List[BaseReticleTask]:
        # compute size for one reticle
        compute_amount = sum([op.get_fp_mac_count() if forward else op.get_bp_mac_count() + op.get_fp_mac_count() 
                              for name, op in self._op_graph.nodes(data='operator')]) / self.micro_batch_size
        compute_amount = compute_amount * self.num_layer_per_pipeline_stage / self.num_reticle_per_model_chunk

        task_list = []

        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            compute_task = ComputeReticleTask(virtual_reticle_id, compute_amount, repeated_times=self.micro_batch_size)
            task_list.append(compute_task)
        
        return task_list
    
    def __assign_allreduce_reticle_task(self, forward: bool):
        tensors = self._op_graph.get_tensors()
        allreduce_numel = sum([tensor.numel() for tensor in tensors.values() if tensor.name in (['Z', 'X2_mlp'] if forward else ['X', 'X0_mlp', 'Z', 'X2_mlp'])])
        allreduce_numel = allreduce_numel * self.num_layer_per_pipeline_stage / self.num_reticle_per_model_chunk / self.micro_batch_size
        allreduce_numel = allreduce_numel * 2 * (self.tensor_parallel_size - 1) / self.tensor_parallel_size
        allreduce_size = allreduce_numel * self.training_config.get_precision_size()

        task_list = []
        
        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            pipeline_stage_index, tensor_parallel_index, worker_reticle_index = parallel_index
            peer_tensor_parallel_index = (tensor_parallel_index + 1) % self.tensor_parallel_size
            peer_parallel_index = (pipeline_stage_index, peer_tensor_parallel_index, worker_reticle_index)
            peer_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id[peer_parallel_index]
            allreduce_task = PeerAccessReticleTask(virtual_reticle_id, peer_virtual_reticle_id, 'read', allreduce_size, repeated_times=self.micro_batch_size)
            task_list.append(allreduce_task)
        
        return task_list
    
    def __run_wse_task(self, wse_task: ListWaferTask) -> float:
        mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
        wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
        return wse_evaluator.get_total_latency()
    
    def get_propagation_latency(self, forward=True, detailed_report=False):
        input_task_list = self.__assign_input_reticle_task(forward)
        need_to_swap_weight, need_to_swap_activation = self.get_sram_utilization(forward)
        swap_weight_task_list = self.__assign_swap_weight_reticle_task(forward) if need_to_swap_weight else []
        swap_activation_task_list = self.__assign_swap_activation_reticle_task(forward) if need_to_swap_activation else []
        compute_task_list = self.__assign_compute_reticle_task(forward)
        allreduce_task_list = self.__assign_allreduce_reticle_task(forward) if self.tensor_parallel_size != 1 else []

        task_lists = {
            'input': input_task_list,
            'swap_weight': swap_weight_task_list,
            'swap_activation': swap_activation_task_list,
            'compute': compute_task_list,
            'allreduce': allreduce_task_list,
        }

        if self.is_overlap:
            task_list = sum(task_lists.values(), [])
            wse_task = ListWaferTask(task_list)
            mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
            wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
            total_latency = wse_evaluator.get_total_latency()
            if detailed_report:
                util_report = wse_evaluator.profile_utilization()
                final_report = {
                    'compute': util_report['compute'] * total_latency,
                    'inter_reticle': util_report['inter_reticle'] * total_latency,
                }
        else:
            raw_report = {}
            for task_type, task_list in task_lists.items():
                raw_report[task_type] = self.__run_wse_task(ListWaferTask(task_list)) if task_list else 0
            final_report = {
                'compute': raw_report['compute'],
                'inter_reticle': raw_report['swap_weight'] + raw_report['swap_activation'] + raw_report['input'] + raw_report['allreduce'],
            }
            logger.debug(raw_report)  # in non-overlapping model, you can see latency of each part
            total_latency = sum(raw_report.values(), 0)

        if detailed_report:
            return final_report
        else:
            return total_latency

    def __create_propagation_reticle_task(self, virtual_reticle_id: int, forward: bool) -> FusedReticleTask:
        logger.warning(f"Method {__name__} is deprecated in the future")
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
        logger.warning(f"Method {__name__} is deprecated in the future")

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

    
    def __get_linear_best_blocking(self, 
                                   shapes: Dict[str, Tuple],
                                   dim_types: Dict[str, Tuple],
                                   M_upper_bound = None,
                                   N_upper_bound = None,
                                   ) -> Tuple[int, int]:
        """ Find best blocking strategy for linear operators with output stationary dataflow.
        
        Args:
            - shapes: shapes of A, B and Y, or other intermeidate tensors
            - dim_types: M, N for splitting, K for reduction, - for placeholder
        Returns:
            #split_M, #split_N
        """
        assert 'M' not in dim_types['B']
        assert 'N' not in dim_types['A']
        assert 'M' in dim_types['Y']
        assert 'N' in dim_types['Y']
        assert 'M' in dim_types['A']
        assert 'N' in dim_types['B']
        for name in shapes:
            assert len(shapes[name]) == len(dim_types[name])

        M_value, N_value = None, None
        for dim_value, dim_type in zip(shapes['Y'], dim_types['Y']):
            if dim_type == 'M': M_value = dim_value
            if dim_type == 'N': N_value = dim_value
        precision = self.training_config.get_precision_size()
        reticle_sram_size = Reticle.get_sram_size(self.wafer_scale_engine.reticle_config)
        
        def get_tensor_size(name):
            return reduce(lambda x, y: x * y, shapes[name]) * precision
        A_size, B_size, Y_size = get_tensor_size('A'), get_tensor_size('B'), get_tensor_size('Y')

        best_split = (M_value, N_value)
        best_dram_access_amount = A_size * N_value + B_size * M_value + Y_size

        def get_buffer_size(name, split_M, split_N):
            def get_dim_buffer_size(dim_value, dim_type):
                table = {
                    'M': dim_value // split_M,
                    'N': dim_value // split_N,
                    'K': 1,
                }
                return table.get(dim_type, dim_value)
            buffer_size = reduce(lambda x, y: x * y,
                                 [get_dim_buffer_size(dim_value, dim_type) 
                                  for dim_value, dim_type in zip(shapes[name], dim_types[name])])
            buffer_size *= precision
            return buffer_size
        
        for split_M, split_N in zip(factoring(M_value, M_upper_bound), factoring(N_value, N_upper_bound)):
            total_buffer_size = reduce(lambda x, y: x + y, 
                                       [get_buffer_size(name, split_M, split_N)
                                        for name in shapes])
            if total_buffer_size <= reticle_sram_size:
                dram_access_amount = A_size * split_N + B_size * split_M + Y_size
                if dram_access_amount < best_dram_access_amount:
                    best_split = (split_M, split_N)
                    best_dram_access_amount = dram_access_amount
        return best_split
    
    

    def __get_forward_propatation_latency(self, store_all_activation=False) -> float:
        """
        Store all activation in rematerialization state
        """
        tensors = self._op_graph.get_tensors()
        precision = self.training_config.get_precision_size()
        self_attention_repeats = {}

        # self attention
        self_attention_repeats['W_qkv'], self_attention_repeats['X'] = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['X'].shape,
                'B': tensors['W_qkv'].shape,
                'Y': tensors['Y'].shape,
                'QKV': tensors['QKV'].shape,
                'scores': tensors['scores'].shape,
            },
            dim_types={
                'A': ('M', '-', 'K'),
                'B': ('K', 'N'),
                'Y': ('M', 'N', '-', '-'),
                'QKV': ('M', '-', 'N'),
                'scores': ('M', 'N', '-', '-'),
            },
            N_upper_bound=self.attention_heads//self.tensor_parallel_size,
        )
        # Proj
        self_attention_repeats['W_proj'], self_attention_repeats['Y'] = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['Y_reshape'].shape,
                'B': tensors['W_proj'].shape,
                'Y': tensors['Z'].shape,
            },
            dim_types={
                'A': ('M', '-', 'K'),
                'B': ('K', 'N'),
                'Y': ('M', '-', 'N'),
            },
        )

        self_attention_read_data_amount = sum([tensors[name].numel() * repeat for name, repeat in self_attention_repeats.items()]) * precision
        self_attention_write_tensor_names = ['Y', 'Z'] if not store_all_activation else ['QKV', 'scores', 'Y', 'Z']
        self_attention_write_data_amount = sum([tensors[name].numel() for name in self_attention_write_tensor_names]) * precision
        self_attention_compute_amount = sum([op.get_fp_mac_count() for node, op in self._op_graph.nodes(data='operator') 
                                            if node in ('linear_qkv', 'matmul_qk', 'matmul_scorev', 'linear_proj')])
        self_attention_task_generator = ThreeStageReticleTaskGenerator(**{
            'compute_amount': self_attention_compute_amount,
            'read_data_amount': [self_attention_read_data_amount],
            'write_data_amount': [self_attention_write_data_amount],
            'reuse_dram_port': False,
        })
        logger.info(f"Running self-attention submodule forward propagation")
        self_attention_wse_task = ListWaferTask([self_attention_task_generator(repeated_times=1) for _ in self.virtual_reticle_id_2_parallel_index])
        self_attention_latency = self.__run_wse_task(self_attention_wse_task)

        mlp_repeats = {}
        # MLP 0
        mlp_repeats['W0_mlp'], mlp_repeats['X0_mlp'] = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['X0_mlp'].shape,
                'B': tensors['W0_mlp'].shape,
                'Y': tensors['X1_mlp'].shape,
            },
            dim_types={
                'A': ('M', '-', 'K'),
                'B': ('K', 'N'),
                'Y': ('M', '-', 'N'),
            },
        )

        # MLP 1
        mlp_repeats['W1_mlp'], mlp_repeats['X1_mlp'] = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['X1_mlp'].shape,
                'B': tensors['W1_mlp'].shape,
                'Y': tensors['X2_mlp'].shape,
            },
            dim_types={
                'A': ('M', '-', 'K'),
                'B': ('K', 'N'),
                'Y': ('M', '-', 'N'),
            },
        )
        mlp_read_data_amount = sum([tensors[name].numel() * repeat for name, repeat in mlp_repeats.items()]) * precision
        mlp_write_data_amount = sum([tensors[name].numel() for name in ('X1_mlp', 'X2_mlp')]) * precision
        mlp_compute_amount = sum([op.get_fp_mac_count() for node, op in self._op_graph.nodes(data='operator') 
                                            if node in ('linear_mlp0', 'linear_mlp1')])
        mlp_task_generator = ThreeStageReticleTaskGenerator(**{
            'compute_amount': mlp_compute_amount,
            'read_data_amount': [mlp_read_data_amount],
            'write_data_amount': [mlp_write_data_amount],
            'reuse_dram_port': False,
        })
        logger.info(f"Running MLP submodule forward propagation")
        mlp_wse_task = ListWaferTask([mlp_task_generator(repeated_times=1) for _ in self.virtual_reticle_id_2_parallel_index])
        mlp_latency = self.__run_wse_task(mlp_wse_task)

        total_latency = (self_attention_latency + mlp_latency) * self.num_layer_per_pipeline_stage

        logger.debug(f"FP self-attention repeats: {self_attention_repeats}")
        logger.debug(f"FP MLP repeats: {mlp_repeats}")

        return total_latency
    
    def __get_backward_propagation_latency(self) -> float:
        tensors = self._op_graph.get_tensors()
        precision = self.training_config.get_precision_size()

        mlp_repeats = {
            'W1_mlp': 0,
            'W0_mlp': 0,
            'X2_mlp': 0,
            'X1_mlp': 0,
            'X0_mlp': 0,
        }  # calculate input grad
        # MLP 2
        split_X1_mlp = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['X2_mlp'].shape,
                'B': tensors['W1_mlp'].shape,
                'Y': tensors['X1_mlp'].shape,
            },
            dim_types={
                'A': ('M', '-', 'K'),
                'B': ('N', 'K'),
                'Y': ('M', '-', 'N'),
            },
        )
        mlp_repeats['W1_mlp'] += split_X1_mlp[0]
        mlp_repeats['X2_mlp'] += split_X1_mlp[1]

        split_W2_mlp = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['X1_mlp'].shape,
                'B': tensors['X2_mlp'].shape,
                'Y': tensors['W1_mlp'].shape,
            },
            dim_types={
                'A': ('K', 'K', 'M'),
                'B': ('K', 'K', 'N'),
                'Y': ('M', 'N'),
            },
        )
        mlp_repeats['X2_mlp'] += split_W2_mlp[0]
        mlp_repeats['X1_mlp'] += split_W2_mlp[1]

        # MLP 1
        split_X0_mlp = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['X1_mlp'].shape,
                'B': tensors['W0_mlp'].shape,
                'Y': tensors['X0_mlp'].shape,
            },
            dim_types={
                'A': ('M', '-', 'K'),
                'B': ('N', 'K'),
                'Y': ('M', '-', 'N'),
            },
        )
        mlp_repeats['W0_mlp'] += split_X0_mlp[0]
        mlp_repeats['X1_mlp'] += split_X0_mlp[1]

        split_W1_mlp = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['X0_mlp'].shape,
                'B': tensors['X1_mlp'].shape,
                'Y': tensors['W0_mlp'].shape,
            },
            dim_types={
                'A': ('K', 'K', 'M'),
                'B': ('K', 'K', 'N'),
                'Y': ('M', 'N'),
            },
        )
        mlp_repeats['X1_mlp'] += split_W1_mlp[0]
        mlp_repeats['X0_mlp'] += split_W1_mlp[1]

        mlp_read_data_amount = sum([tensors[name].numel() * repeat for name, repeat in mlp_repeats.items()]) * precision
        mlp_write_data_amount = sum([tensors[name].numel() for name in ('X1_mlp', 'X0_mlp', 'W1_mlp', 'W0_mlp')]) * precision
        mlp_compute_amount = sum([op.get_bp_mac_count() for node, op in self._op_graph.nodes(data='operator') 
                                            if node in ('linear_mlp0', 'linear_mlp1')])
        mlp_task_generator = ThreeStageReticleTaskGenerator(**{
            'compute_amount': mlp_compute_amount,
            'read_data_amount': [mlp_read_data_amount],
            'write_data_amount': [mlp_write_data_amount],
            'reuse_dram_port': False,
        })
        logger.info(f"Running MLP module backward propagation")
        mlp_wse_task = ListWaferTask([mlp_task_generator(repeated_times=1) for _ in self.virtual_reticle_id_2_parallel_index])
        mlp_latency = self.__run_wse_task(mlp_wse_task)

        self_attention_repeats = {
            'Z': 0,
            'W_proj': 0,
            'Y_reshape': 0,
            'Y': 1,  # fetch 1 Y's grad
            'V': 1,  # fetch 1 score and calc 1 V's grad, store
            'scores': 1,  # fetch 1 V and calc 1 score's grad, needless to store
            'Q': 1,  # fetch 1 K and calc 1 Q's grad
            'K': 1,  # fetch 1 Q and calc 1 K's grad
            'QKV': 0,  # and QKV's grad is written once! It can then be fetched multiple times
            'W_qkv': 0,
            'X': 0,
        }
        # Z = W_proj(Y)
        split_Y_reshape = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['Z'].shape,
                'B': tensors['W_proj'].shape,
                'Y': tensors['Y_reshape'].shape,
            },
            dim_types={
                'A': ('M', '-', 'K'),
                'B': ('N', 'K'),
                'Y': ('M', '-', 'N'),
            },
        )
        self_attention_repeats['W_proj'] += split_Y_reshape[0]
        self_attention_repeats['Z'] += split_Y_reshape[1]

        split_W_proj = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['Y_reshape'].shape,
                'B': tensors['Z'].shape,
                'Y': tensors['W_proj'].shape,
            },
            dim_types={
                'A': ('K', 'K', 'M'),
                'B': ('K', 'K', 'N'),
                'Y': ('M', 'N'),
            },
        )
        self_attention_repeats['Z'] += split_W_proj[0]
        self_attention_repeats['Y_reshape'] += split_W_proj[1]

        # QKV = W_qkv(X)
        split_X = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['QKV'].shape,
                'B': tensors['W_qkv'].shape,
                'Y': tensors['X'].shape,
            },
            dim_types={
                'A': ('M', '-', 'K'),
                'B': ('N', 'K'),
                'Y': ('M', '-', 'N'),
            },
        )
        self_attention_repeats['W_qkv'] += split_X[0]
        self_attention_repeats['QKV'] += split_X[1]


        split_W_qkv = self.__get_linear_best_blocking(
            shapes={
                'A': tensors['X'].shape,
                'B': tensors['QKV'].shape,
                'Y': tensors['W_qkv'].shape,
            },
            dim_types={
                'A': ('K', 'K', 'M'),
                'B': ('K', 'K', 'N'),
                'Y': ('M', 'N'),
            },
        )
        self_attention_repeats['QKV'] += split_W_qkv[0]
        self_attention_repeats['X'] += split_W_qkv[1]

        self_attention_read_data_amount = sum([tensors[name].numel() * repeat for name, repeat in self_attention_repeats.items()]) * precision
        self_attention_write_data_amount = sum([tensors[name].numel() for name in ('W_proj', 'Y_reshape', 'QKV', 'W_qkv', 'X')]) * precision
        self_attention_compute_amount = sum([op.get_bp_mac_count() for node, op in self._op_graph.nodes(data='operator') 
                                            if node in ('linear_qkv', 'matmul_qk', 'matmul_scorev', 'linear_proj')])
        self_attention_task_generator = ThreeStageReticleTaskGenerator(**{
            'compute_amount': self_attention_compute_amount,
            'read_data_amount': [self_attention_read_data_amount],
            'write_data_amount': [self_attention_write_data_amount],
            'reuse_dram_port': False,
        })
        logger.info(f"Running self-attention submodule backward propagation")
        self_attention_wse_task = ListWaferTask([self_attention_task_generator(repeated_times=1) for _ in self.virtual_reticle_id_2_parallel_index])
        self_attention_latency = self.__run_wse_task(self_attention_wse_task)

        rematerialization_latency = self.__get_forward_propatation_latency(store_all_activation=True)

        # this part is ignored, since weight shouldn't be updated for every micro batch
        # for every mini-batch, it's completely transmission bounded

        # weight_total_numel = sum([tensor.numel() for tensor in tensors.values() if tensor.kind == 'weight'])
        # optimizer_state_factor = self.training_config.get_optimizer_state_size()
        # weight_update_compute_factor = self.training_config.get_weight_update_compute_amount()
        # weight_update_dram_access_amount = weight_total_numel * (precision + optimizer_state_factor)
        # weight_update_compute_amount = weight_total_numel * weight_update_compute_factor
        # weight_update_task_generator = ThreeStageReticleTaskGenerator(**{
        #     'compute_amount': weight_update_compute_amount,
        #     'read_data_amount': [weight_update_dram_access_amount],
        #     'write_data_amount': [weight_update_dram_access_amount],
        #     'reuse_dram_port': False,
        # })
        # logger.info(f"Running weight updating")
        # weight_update_wse_task = ListWaferTask([weight_update_task_generator(repeated_times=1) for _ in self.virtual_reticle_id_2_parallel_index])
        # weight_update_latency = self.__run_wse_task(weight_update_wse_task)

        total_latency = (mlp_latency + self_attention_latency) * self.num_layer_per_pipeline_stage + rematerialization_latency

        logger.debug(f"BP self-attention repeats: {self_attention_repeats}")
        logger.debug(f"BP MLP repeats: {mlp_repeats}")

        return total_latency
