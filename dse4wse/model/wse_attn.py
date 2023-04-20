
from typing import Dict, Tuple, Union, List
from math import ceil
from itertools import product
from functools import reduce
import networkx as nx
import numpy as np
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
        self.num_reticle_per_model_chunk = wafer_scale_engine.reticle_array_height * wafer_scale_engine.reticle_array_width // self.tensor_parallel_size
        # check valid model chunking
        assert number_of_layers % model_parallel_size == 0
        assert hidden_size % tensor_parallel_size == 0

        # intra-model-chunk params, transparent to users
        # Best exec params differ from training to inference, so we find optimal setting
        # everytime we call on perf API.
        self.weight_streaming = None
        self.nano_batch_size = None
        self.layer_pipeline_size = None
        self.weight_swapping_factor = None

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
        # check available reticle
        used_reticle = self.tensor_parallel_size * self.num_reticle_per_model_chunk
        total_reticle = self.wafer_scale_engine.reticle_array_height * self.wafer_scale_engine.reticle_array_width
        assert used_reticle <= total_reticle, f"Requiring {used_reticle} reticles but there's only {total_reticle}"
        self.num_layer_per_model_chunk = (self.number_of_layers // self.model_parallel_size)

        # training settings 
        self.training_config = training_config
    
    def _find_best_intra_model_chunk_exec_params(self, inference: bool) -> None:
        """ Compiler find the best parameters to execute one model chunk

        First, it tries layer pipelining mode. 
        We intend to use layer pipelining for better resource utilization,
        however, in current setup, layer pipelining only degradate perf due to pipeline bubbles.
        Then, when weight cannot be held on SRAM, it uses weight streaming mode.
        We try to keep as much activation on chip as possible, however, when the activation
        of single sequence cannot be withheld, we swap weight more often.
        """

        # reset exec params
        self.weight_streaming = None
        self.nano_batch_size = None
        self.layer_pipeline_size = None
        self.weight_swapping_factor = None

        # try layer pipelining, we try to minimize bubble, so layer pipeline stage stays at 1
        self.layer_pipeline_size = 1
        self.weight_swapping_factor = 1

        # try layer pipeline, then weight streaming
        for weight_streaming in [False, True]:
            self.weight_streaming = weight_streaming
            for nano_batch_size in sorted(factoring(self.micro_batch_size), reverse=True):
                try: assert self.get_sram_utilization(inference, nano_batch_size, self.layer_pipeline_size, self.weight_streaming)
                except AssertionError: continue
                self.nano_batch_size = nano_batch_size
                break
            if self.nano_batch_size:
                assert self.weight_streaming is not None
                assert self.nano_batch_size is not None
                assert self.layer_pipeline_size is not None
                assert self.weight_swapping_factor is not None
                # logger.debug(f"Found valid nano batch size = {self.nano_batch_size}")
                # logger.debug(f"weight streaming: {self.weight_streaming}")
                # logger.debug(f"weight of model chunk: {self._get_weight_numel_per_model_chunk() * self.training_config.get_precision_size() / 1e6} MB")
                # logger.debug(f"sram usage = {self._get_sram_usage(inference, self.nano_batch_size, self.layer_pipeline_size, self.weight_streaming) / 1e6} MB")
                # logger.debug(f"sram size = {Reticle.get_sram_size(self.wafer_scale_engine.reticle_config) * self.num_reticle_per_model_chunk / 1e6} MB")
                # logger.debug(f"can it do layer pipelining? {self.get_sram_utilization(inference, self.nano_batch_size, 1, False)}")
                return

        # swap more weight when 1 seq cannot be withheld
        self.nano_batch_size = 1
        act_size = self._get_sram_usage(inference, nano_batch_size=1, layer_pipeline_size=1, weight_streaming=True)
        sram_size = Reticle.get_sram_size(self.wafer_scale_engine.reticle_config) * self.num_reticle_per_model_chunk
        assert act_size > sram_size
        self.weight_swapping_factor = ceil(act_size / sram_size)

        assert self.weight_streaming is not None
        assert self.nano_batch_size is not None
        assert self.layer_pipeline_size is not None
        assert self.weight_swapping_factor is not None
        # logger.debug(f"Found valid nano batch size = {self.nano_batch_size}")
        # logger.debug(f"weight streaming: {self.weight_streaming}")
        # logger.debug(f"weight of model chunk: {self._get_weight_numel_per_model_chunk() * self.training_config.get_precision_size() / 1e6} MB")
        # logger.debug(f"sram usage = {self._get_sram_usage(inference, self.nano_batch_size, self.layer_pipeline_size, self.weight_streaming)}")
        # logger.debug(f"sram size = {Reticle.get_sram_size(self.wafer_scale_engine.reticle_config) * self.num_reticle_per_model_chunk}")
        # logger.debug(f"can it do layer pipelining? {self.get_sram_utilization(inference, self.nano_batch_size, 1, False)}")

    # helper functions for API

    def _get_flops_per_model_chunk_per_seq_per_layer(self, inference=False):
        """ Calculate FLOPS, copied from Cerebras-GPT appendix.
        Embedding and final logits are ignored for simplicity.
        """
        d_model = self.hidden_size
        seq_len = self.sequence_length
        num_heads = self.attention_heads // self.tensor_parallel_size
        key_size = d_model // self.attention_heads

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
            return total_flops * 3
    
    def _get_ideal_compute_latency(self, inference: bool) -> float:
        """ Ideal compute latency of a model chunk
        Assume 100% compute resource utilization.
        """
        assert self.zero_dp_p is False

        compute_power = self.wafer_scale_engine.reticle_compute_power * self.num_reticle_per_model_chunk
        total_flops = self._get_flops_per_model_chunk_per_seq_per_layer(inference) * self.micro_batch_size * self.num_layer_per_model_chunk
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
        kq_logits = softmax = (seq_len ** 2) * (num_heads)
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
        """ Fetching input activation / grad onto the wafer

        Weight streaming: 
        - fp: load 1 micro batch size input
        - bp: load 1 micro batch size grad, and activation checkpoints of every layer
        Layer pipelining: 
        - fp: load 1 micro batch size input
        - bp: load 1 micro batch size input & grad
        """
        input_tensor_numel = self.micro_batch_size * self.sequence_length * self.hidden_size
        if self.weight_streaming:
            if not inference: input_tensor_numel *= (1 + self.num_layer_per_model_chunk)
        else:
            if not inference: input_tensor_numel *= 2

        dram_access_size = input_tensor_numel * self.training_config.get_precision_size()
        dram_access_latency = dram_access_size / self.wafer_scale_engine.dram_bandwidth

        inter_reticle_size = input_tensor_numel * self.training_config.get_precision_size() * (self.tensor_parallel_size - 1)
        bisection_bandwidth = self.wafer_scale_engine.get_bisection_bandwidth()
        inter_reticle_latency = inter_reticle_size / bisection_bandwidth

        total_latency = dram_access_latency + inter_reticle_latency
        return total_latency
    
    def _get_output_latency(self, inference: bool) -> float:
        """ Writing activation / grad back to DRAM

        Weight streaming: 
        - fp: write every layer's output, but we count this part in bp
        - bp: write 1 micro batch size grad
        Layer pipelining:
        - fp: write 1 micro batch output
        - bp: write 1 micro batch grad 
        """
        output_tensor_numel = self.micro_batch_size * self.sequence_length * self.hidden_size
        if self.weight_streaming:
            if not inference: output_tensor_numel *= self.num_layer_per_model_chunk

        dram_access_size = inter_reticle_size = output_tensor_numel * self.training_config.get_precision_size()
        dram_access_latency = dram_access_size / self.wafer_scale_engine.dram_bandwidth
        bisection_bandwidth = self.wafer_scale_engine.get_bisection_bandwidth()
        inter_reticle_latency = inter_reticle_size / bisection_bandwidth

        total_latency = dram_access_latency + inter_reticle_latency
        return total_latency

    def _get_swap_weight_latency(self, inference: bool) -> float:
        """
        Estimation on swapping weight in and out of DRAM, only for weight streaming mode
        """
        if not self.weight_streaming:
            return 0
        assert self.nano_batch_size, "You should set nano batch size first"

        weight_numel_per_model_chunk = self._get_weight_numel_per_model_chunk()
        weight_numel_per_wafer = weight_numel_per_model_chunk * self.tensor_parallel_size
        swapped_weight_numel = ceil(self.micro_batch_size / self.nano_batch_size) * weight_numel_per_wafer
        if not inference: swapped_weight_numel *= 3  # fp-read + rebuild + bp-read + bp write grad, we fuse rebuild and bp-read for simplicity
        swapped_weight_numel *= self.weight_swapping_factor
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
    
    def _get_sram_usage(self, inference: bool, nano_batch_size: int = None, layer_pipeline_size: int = None, weight_streaming: bool = None) -> int:
        precision = self.training_config.get_precision_size()
        if nano_batch_size is None: nano_batch_size = self.nano_batch_size
        if layer_pipeline_size is None: layer_pipeline_size = self.layer_pipeline_size
        if weight_streaming is None: weight_streaming = self.weight_streaming
        assert nano_batch_size is not None
        assert layer_pipeline_size is not None
        assert weight_streaming is not None

        attn_act_size = self._get_attn_activation_numel_per_model_chunk_per_seq_per_layer() * precision
        ffn_act_size = self._get_ffn_activation_numel_per_model_chunk_per_seq_per_layer() * precision
        if weight_streaming:
            assert layer_pipeline_size == 1, "You shouldn't set layer pipelining on weight streaming mode!"
            if inference:
                # only store 1 layer's activation
                act_size = max(attn_act_size, ffn_act_size) * nano_batch_size
            else:
                # buffer all activations -- is impossible for very large models
                # we allow rematerialize activation layer by layer
                # in this way, we maximize reuse of weight
                act_size = (attn_act_size + ffn_act_size) * nano_batch_size
            # logger.debug(f"nano batch size: {nano_batch_size}, weight streaming act size: {act_size / 1e9} GB, sram size: {sram_size / 1e9} GB")
            return act_size  # save some space for weight
        else:  # layer pipelining
            weight_size = self._get_weight_numel_per_model_chunk()
            # each pipeline stage hold one complete share of activation
            act_size = layer_pipeline_size * nano_batch_size * (attn_act_size + ffn_act_size)
            if inference:
                return weight_size + act_size
            else:
                # each layer pipeline stage need to buffer the same number of nano batch as the depth of pipeline
                if layer_pipeline_size == 1:
                    checkpoint_size = 0
                else:
                    checkpoint_size = layer_pipeline_size * (layer_pipeline_size * nano_batch_size) * self.sequence_length * self.hidden_size
                return checkpoint_size + weight_size + act_size

    
    def get_sram_utilization(self, inference: bool, nano_batch_size: int = None, layer_pipeline_size: int = None, weight_streaming: bool = None) -> bool:
        """ Check if SRAM could hold a model chunk
        """
        sram_size = Reticle.get_sram_size(self.wafer_scale_engine.reticle_config) * self.num_reticle_per_model_chunk
        sram_usage = self._get_sram_usage(inference, nano_batch_size, layer_pipeline_size, weight_streaming)
        return sram_usage < sram_size
    
    def get_propagation_latency(self, inference: bool, detailed_report=False):
        """ Propagation latency of a micro batch, 1F for inference, 1F1B for training.
        Naive version doesn't consider overlapping, but lp_solver version does
        """
        input_latency = self._get_input_latency(inference=inference)
        swap_weight_latency = self._get_swap_weight_latency(inference=inference)
        compute_latency = self._get_compute_latency(inference=inference)
        activation_allreduce_latency = self._get_activation_allreduce_latency(inference=inference)
        output_latency = self._get_output_latency(inference=inference)

        total_latency = compute_latency + activation_allreduce_latency + swap_weight_latency + input_latency + output_latency
        idle_latency = (self.layer_pipeline_size - 1) / (self.micro_batch_size // self.nano_batch_size) * total_latency
        total_latency += idle_latency

        # debugging
        raw_report = {
            'input': input_latency,
            'swap_weight': swap_weight_latency,
            'compute': compute_latency,
            'allreduce': activation_allreduce_latency,
            'output': output_latency,
        }
        logger.debug(raw_report)

        if detailed_report:
            return {
                'compute': compute_latency,
                'inter_reticle': input_latency + activation_allreduce_latency + swap_weight_latency + output_latency,
                'idle': idle_latency,
            }
        else:
            return total_latency
    
    # API for evaluations

    def get_dram_utilization(self) -> bool:
        """ This API only considers training DRAM utilization.
        Since DRAM is assumed to be infinite, this API is no longer useful
        """
        # static memory
        # weight & grad & optimizer state
        assert self.zero_dp_p is False

        # weight / grad / optimizer state on the wafer
        weight_tensor_numel = self._get_weight_numel_per_model_chunk() * self.tensor_parallel_size
        weight_tensor_size = weight_tensor_numel * self.training_config.get_precision_size()
        weight_grad_tensor_size = weight_tensor_numel * self.training_config.get_precision_size()
        if self.zero_dp_g: weight_grad_tensor_size /= self.data_parallel_size
        weight_optimizer_state_tensor_size = weight_tensor_numel * self.training_config.get_optimizer_state_size()
        if self.zero_dp_os: weight_optimizer_state_tensor_size /= self.data_parallel_size

        # input checkpoint for model parallel, 1F1B
        checkpoint_tensor_numel = self.model_parallel_size * self.micro_batch_size * self.sequence_length * self.hidden_size
        if self.weight_streaming: checkpoint_tensor_numel *= self.num_layer_per_model_chunk
        checkpoint_tensor_size = checkpoint_tensor_numel * self.training_config.get_precision_size()

        # activation tensor no longer goes to DRAM

        utilized_memory = weight_tensor_size + weight_grad_tensor_size + weight_optimizer_state_tensor_size + checkpoint_tensor_size
        total_memory = self.wafer_scale_engine.dram_size

        logger.info("Summary for checking DRAM utilization:")
        logger.info(f"weight                  : {int(weight_tensor_size / 1e9):>15d} GB ({weight_tensor_size / total_memory:.2%})")
        logger.info(f"weight's grad           : {int(weight_grad_tensor_size / 1e9):>15d} GB ({weight_grad_tensor_size/ total_memory:.2%})")
        logger.info(f"weight's optimizer state: {int(weight_optimizer_state_tensor_size / 1e9):>15d} GB ({weight_optimizer_state_tensor_size/ total_memory:.2%})")
        logger.info(f"checkpoint              : {int(checkpoint_tensor_size / 1e9):>15d} GB ({checkpoint_tensor_size / total_memory:.2%})")
        logger.info(f"Total utilized DRAM     : {int(utilized_memory / 1e9):>15d} GB ({utilized_memory / total_memory:.2%})")

        return utilized_memory <= total_memory
    
    def get_training_throughput(self) -> float:
        """ Use GPipe's pipelining technique.
        """
        logger.info("Calculating training throughput of attention module")

        self._find_best_intra_model_chunk_exec_params(inference=False)

        single_stage_latency = self.get_propagation_latency(inference=False) + self._get_activation_cross_pipeline_stage_latency() * 2
        weight_comm_latency = self._get_weight_update_latency()
        pipeline_factor = (self.model_parallel_size - 1) + ceil(self.mini_batch_size / self.micro_batch_size / self.data_parallel_size)
        whole_batch_latency = pipeline_factor * single_stage_latency + weight_comm_latency
        sequence_per_sec = self.mini_batch_size / whole_batch_latency

        return sequence_per_sec

    def get_training_wse_utilization(self) -> float:
        logger.info("Calculating training wse utilization of attention module")

        self._find_best_intra_model_chunk_exec_params(inference=False)
        
        single_stage_latency = self.get_propagation_latency(inference=False) + self._get_activation_cross_pipeline_stage_latency() * 2
        weight_update_latency = self._get_weight_update_latency() 
        pipeline_factor = (self.model_parallel_size - 1) + ceil(self.mini_batch_size / self.micro_batch_size / self.data_parallel_size)
        whole_batch_latency = pipeline_factor * single_stage_latency + weight_update_latency

        ideal_compute_latency = self._get_ideal_compute_latency(inference=False)
        ideal_compute_latency *= ceil(self.mini_batch_size / self.micro_batch_size / self.data_parallel_size)
        utilization = ideal_compute_latency / whole_batch_latency

        # debugging
        bp_report = self.get_propagation_latency(inference=False, detailed_report=True)
        num_micro_batch = ceil(self.mini_batch_size / self.micro_batch_size / self.data_parallel_size)
        full_report = {
            'compute': num_micro_batch * bp_report['compute'],
            'inter_reticle': num_micro_batch * bp_report['inter_reticle'],
            'inter_wafer': num_micro_batch * (self._get_activation_cross_pipeline_stage_latency() * 2) + weight_update_latency,
            'idle': (self.model_parallel_size - 1) * single_stage_latency + num_micro_batch * bp_report['idle'],
        }
        logger.info("Summary for checking time utilization proportion:")
        for k, v in full_report.items():
            logger.info(f"{k:<20} : {int(v*1e3):>15d} ms ({v/whole_batch_latency:.2%})")

        return utilization
    
    # TODO: inference should calculate less computation and more memory access

    def get_inference_latency(self) -> float:
        logger.info("Calculating inference latency of attention module")

        self._find_best_intra_model_chunk_exec_params(inference=True)

        single_stage_fp_latency = self.get_propagation_latency(inference=True)
        total_fp_latency = single_stage_fp_latency * self.model_parallel_size
        total_cross_wafer_times = self.model_parallel_size + 1  # in, cross_wafer, out
        cross_wafer_latency = self._get_activation_cross_pipeline_stage_latency(connect_type='wafer') * total_cross_wafer_times

        total_latency = total_fp_latency + cross_wafer_latency
        return total_latency

    def get_training_peak_power(self) -> float:
        """ Peak performance during 1 propagation.
        We can provide this API in high fidelity much more easily
        """
        raise NotImplementedError
    
class ReticleFidelityWseTransformerRunner(WseTransformerRunner):
    """ Estimate WSE performance with higher fidelity
    Propagation latency now considers more inter-reticle transmission features: 
    - Traffic induced by accessing DRAM
    - Congestion in allreduce induced by inappropiate mapping
    - Overlapping computation and inter-reticle communication
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
                 **kwargs,
                 ) -> None:
        super().__init__(attention_heads, hidden_size, sequence_length, number_of_layers, micro_batch_size, mini_batch_size, data_parallel_size, model_parallel_size, tensor_parallel_size, wafer_scale_engine, training_config, inter_wafer_bandwidth, zero_dp_os, zero_dp_g, zero_dp_p, zero_r_pa, **kwargs)
        self.num_pipeline_stage_per_wafer = 1   # stay compatible with old codes
        self.virtual_reticle_id_2_parallel_index = {i: (m, t, r) for i, (m, t, r) in enumerate(product(range(self.num_pipeline_stage_per_wafer), range(tensor_parallel_size), range(self.num_reticle_per_model_chunk)))}
        self.parallel_index_2_virtual_reticle_id = {(m, t, r): i for i, (m, t, r) in enumerate(product(range(self.num_pipeline_stage_per_wafer), range(tensor_parallel_size), range(self.num_reticle_per_model_chunk)))}
        self.is_overlap = True  # if ture, we overlap compute with inter reticle communication
        self.__virtual_dram_port_counter = 0
    
    def __alloc_new_dram_port(self):
        vdp = deepcopy(self.__virtual_dram_port_counter)
        self.__virtual_dram_port_counter += 1
        return vdp
    
    # TODO: rewrite this part with allgather
    # of course this version is fine, so don't worry
    
    def _assign_input_reticle_task(self, inference: bool, rematerialization=False) -> List[BaseReticleTask]:
        """ Reticle tasks for input
        """
        input_tensor_numel = self.nano_batch_size * self.sequence_length * self.hidden_size
        input_tensor_size = input_tensor_numel * self.training_config.get_precision_size()
        if rematerialization and self.weight_streaming: input_tensor_size *= self.num_layer_per_model_chunk
        # rematerialization might use different DRAM port

        repeated_times = ceil(self.micro_batch_size / self.nano_batch_size)

        def need_to_access_dram(worker_reticle_index) -> bool:
            if inference:
                return worker_reticle_index == 0
            else:
                return worker_reticle_index == self.num_reticle_per_model_chunk - 1

        task_list = []

        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            pipeline_stage_index, tensor_parallel_index, worker_reticle_index = parallel_index
            if need_to_access_dram(worker_reticle_index=worker_reticle_index):  # need to fetch input from DRAM
                if tensor_parallel_index == 0:  # only the first worker access DRAM
                    virtual_dram_port = self.__alloc_new_dram_port()
                    reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'read', input_tensor_size, repeated_times=repeated_times)
                    task_list.append(reticle_task)
                else:  # other workers fetch from their peers, forming a binary tree
                    pred_tensor_parallel_index = tensor_parallel_index - 1  # but that would make transmission overlap, and we simply assumes 100% bandwidth utilization
                    pred_parallel_index = (pipeline_stage_index, pred_tensor_parallel_index, worker_reticle_index)
                    pred_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id[pred_parallel_index]
                    reticle_task = PeerAccessReticleTask(virtual_reticle_id, pred_virtual_reticle_id, 'read', input_tensor_size, repeated_times=repeated_times)
                    task_list.append(reticle_task)
            else:  # other workers uses predcessor's activation
                pred_shift = -1 if inference else 1
                pred_worker_reticle_index = worker_reticle_index + pred_shift
                pred_parallel_index = (pipeline_stage_index, tensor_parallel_index, pred_worker_reticle_index)
                pred_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id[pred_parallel_index]
                reticle_task = PeerAccessReticleTask(virtual_reticle_id, pred_virtual_reticle_id, 'read', input_tensor_size, repeated_times=repeated_times)
                task_list.append(reticle_task)

        # rematerialization input tasks
        if inference is False:
            task_list += self._assign_input_reticle_task(inference=True, rematerialization=True)

        return task_list
    
    def _assign_output_reticle_task(self, inference: bool, rematerialization=False) -> List[BaseReticleTask]:
        output_tensor_numel = self.nano_batch_size * self.sequence_length * self.hidden_size
        output_tensor_size = output_tensor_numel * self.training_config.get_precision_size()
        if rematerialization and self.weight_streaming: output_tensor_size *= self.num_layer_per_model_chunk

        repeated_times = ceil(self.micro_batch_size / self.nano_batch_size)

        def need_to_access_dram(worker_reticle_index) -> bool:
            if inference:
                return worker_reticle_index == self.num_reticle_per_model_chunk - 1
            else:
                return worker_reticle_index == 0

        task_list = []

        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            pipeline_stage_index, tensor_parallel_index, worker_reticle_index = parallel_index
            if need_to_access_dram(worker_reticle_index=worker_reticle_index):  # need to write output to DRAM
                if tensor_parallel_index == 0:  # only the first worker access DRAM
                    virtual_dram_port = self.__alloc_new_dram_port()
                    reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'write', output_tensor_size, repeated_times=repeated_times)
                    task_list.append(reticle_task)
                else:  # other workers have the same output, and skip
                    continue
            else:  # other workers uses predcessor's activation -- counted in input reticle task
                continue

        # rematerialization output tasks
        if inference is False:
            task_list += self._assign_output_reticle_task(inference=True, rematerialization=True)

        return task_list
    
    def __assign_swap_weight_reticle_task(self, inference: bool) -> List[BaseReticleTask]:
        if not self.weight_streaming:
            return []

        # weight size for one reticle
        # split all weight evenly (this may not lead to a valid allocation)
        weight_numel_per_model_chunk = self._get_weight_numel_per_model_chunk()
        swapped_weight_numel = weight_numel_per_model_chunk / self.num_reticle_per_model_chunk
        if not inference: swapped_weight_numel *= 3  # fp-read + rebuild + bp-read + bp write grad, we fuse rebuild and bp-read for simplicity
        swapped_weight_numel *= self.weight_swapping_factor
        swapped_weight_size = swapped_weight_numel * self.training_config.get_precision_size()

        repeated_times = ceil(self.micro_batch_size / self.nano_batch_size)
        
        task_list = []

        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            virtual_dram_port = self.__alloc_new_dram_port()
            fp_reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'read', swapped_weight_size, repeated_times=repeated_times)
            task_list.append(fp_reticle_task)
            if not inference:
                # write grad back to DRAM
                bp_reticle_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, 'write', swapped_weight_size, repeated_times=repeated_times)
                task_list.append(bp_reticle_task)

        return task_list

    def __assign_compute_reticle_task(self, inference: bool) -> List[BaseReticleTask]:
        # compute size for one reticle (consider rematerialization seperately)
        compute_amount_per_model_chunk = self._get_flops_per_model_chunk_per_seq_per_layer(inference=inference)
        compute_amount_per_model_chunk *= self.nano_batch_size * self.num_layer_per_model_chunk
        # split compute evenly (this may not lead to a valid allocation)
        compute_amount_per_reticle = compute_amount_per_model_chunk / self.num_reticle_per_model_chunk

        repeated_times = ceil(self.micro_batch_size / self.nano_batch_size)

        task_list = []

        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            compute_task = ComputeReticleTask(virtual_reticle_id, compute_amount_per_reticle, repeated_times=repeated_times)
            task_list.append(compute_task)
        
        return task_list
    
    def __assign_allreduce_reticle_task(self, inference: bool):
        if self.tensor_parallel_size == 1:
            return []
        
        repeated_times = ceil(self.micro_batch_size / self.nano_batch_size)

        allreduce_numel_per_model_chunk = self.nano_batch_size * self.sequence_length * self.hidden_size * 2 * self.num_layer_per_model_chunk  # attn, ffn
        # split allreduce evenly (this definitely does not lead to a valid allocation)
        if not inference: allreduce_numel_per_model_chunk *= 2
        allreduce_numel_per_reticle = allreduce_numel_per_model_chunk / self.num_reticle_per_model_chunk
        allreduce_numel = allreduce_numel_per_reticle * 2 * (self.tensor_parallel_size - 1) / self.tensor_parallel_size  # normalize to single reticle amount
        allreduce_size = allreduce_numel * self.training_config.get_precision_size()

        task_list = []
        
        for virtual_reticle_id, parallel_index in self.virtual_reticle_id_2_parallel_index.items():
            pipeline_stage_index, tensor_parallel_index, worker_reticle_index = parallel_index
            peer_tensor_parallel_index = (tensor_parallel_index + 1) % self.tensor_parallel_size
            peer_parallel_index = (pipeline_stage_index, peer_tensor_parallel_index, worker_reticle_index)
            peer_virtual_reticle_id = self.parallel_index_2_virtual_reticle_id[peer_parallel_index]
            allreduce_task = PeerAccessReticleTask(virtual_reticle_id, peer_virtual_reticle_id, 'read', allreduce_size, repeated_times=repeated_times)
            task_list.append(allreduce_task)
        
        return task_list
    
    def __run_wse_task(self, wse_task: ListWaferTask) -> float:
        mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
        wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
        return wse_evaluator.get_total_latency()

    def _get_task_lists(self, inference: bool) -> Dict[str, List]:
        input_task_list = self._assign_input_reticle_task(inference=inference)
        output_task_list = self._assign_output_reticle_task(inference=inference)
        swap_weight_task_list = self.__assign_swap_weight_reticle_task(inference=inference)
        compute_task_list = self.__assign_compute_reticle_task(inference=inference)
        allreduce_task_list = self.__assign_allreduce_reticle_task(inference=inference)

        task_lists = {
            'input': input_task_list,
            'output': output_task_list,
            'swap_weight': swap_weight_task_list,
            'compute': compute_task_list,
            'allreduce': allreduce_task_list,
        }
        return task_lists

    # API for evaluation

    def get_propagation_latency(self, inference: bool, detailed_report=False):
        task_lists = self._get_task_lists(inference)
        
        if self.is_overlap:
            task_list = sum(task_lists.values(), [])
            wse_task = ListWaferTask(task_list)
            mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
            wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
            total_latency = wse_evaluator.get_total_latency()
            idle_latency = (self.layer_pipeline_size - 1) / (self.micro_batch_size // self.nano_batch_size) * total_latency
            if detailed_report:
                util_report = wse_evaluator.profile_utilization()
                final_report = {
                    'compute': util_report['compute'] * total_latency,
                    'inter_reticle': util_report['inter_reticle'] * total_latency,
                    'idle': idle_latency,
                }
            total_latency += idle_latency
        else:
            raw_report = {}
            for task_type, task_list in task_lists.items():
                raw_report[task_type] = self.__run_wse_task(ListWaferTask(task_list)) if task_list else 0
            logger.debug(raw_report)  # in non-overlapping model, you can see latency of each part
            total_latency = sum(raw_report.values(), 0)
            idle_latency = (self.layer_pipeline_size - 1) / (self.micro_batch_size // self.nano_batch_size) * total_latency
            total_latency += idle_latency
            final_report = {
                'compute': raw_report['compute'],
                'inter_reticle': raw_report['swap_weight'] + raw_report['output'] + raw_report['input'] + raw_report['allreduce'],
                'idle': idle_latency,
            }

        if detailed_report:
            return final_report
        else:
            return total_latency
        
    def get_training_peak_power(self) -> float:
        logger.info("Calculating training peak power of transformer layer")

        self._find_best_intra_model_chunk_exec_params(inference=False)

        task_lists = self._get_task_lists(inference=False)

        # get total latency
        task_list = sum(task_lists.values(), [])
        wse_task = ListWaferTask(task_list)
        mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
        wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
        total_latency = wse_evaluator.get_total_latency()  # in peak power, we ignore layer pipeline bubble

        # debug
        # wse_evaluator.profile_utilization()

        # get total payload
        payload_per_module_type = wse_evaluator.get_module_payload()
        compute_payload = payload_per_module_type['compute']
        interconnect_payload = payload_per_module_type['inter_reticle']
        dram_payload = payload_per_module_type['dram']

        core_num_mac = self.wafer_scale_engine.reticle_config['core_config']['core_compute_power'] / 1e9
        arithmetic_intensity = (core_num_mac / 3) ** 0.5
        sram_payload = compute_payload / arithmetic_intensity

        # get power table
        power_table = self.wafer_scale_engine.buiid_power_table()

        compute_power = power_table.get_compute_power(compute_payload, total_latency)
        interconnect_power = power_table.get_interconnect_power(interconnect_payload, total_latency)
        dram_power = power_table.get_dram_access_power(dram_payload, total_latency)
        sram_power = power_table.get_sram_access_power(sram_payload, total_latency)

        total_power = compute_power + interconnect_power + sram_power + dram_power

        logger.debug(f"compute_power: {compute_power} W ({compute_power / total_power:.2%})")
        logger.debug(f"interconnect_power: {interconnect_power} W ({interconnect_power / total_power:.2%})")
        logger.debug(f"dram_power: {dram_power} W ({dram_power / total_power:.2%})")
        logger.debug(f"sram_power: {sram_power} W ({sram_power / total_power:.2%})")
        logger.debug(f"total power: {total_power} W")

        return total_power