
import os
import sys
import pandas as pd
import numpy as np
import math

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Container, Union
from itertools import chain, combinations, combinations_with_replacement
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import (
    ArchConfig, SbpSignature, TensorInfo, calc_comm_cost_on_same_devices,
    derive_output_sbp_signatures, logger, get_local_tensor_info, TrainingConfig,
    SplitSbpParallel, BroadcastSbpParallel
)

class BaseOperator(ABC):
    """Base class of Operators
    An operator is executed on a logical device constituted by reticles and cores,
    that have local SRAM and 2d-mesh interconnection.
    """

    def __init__(self, 
                name: str, 
                op_type: str, 
                input_tensors: Dict[str, TensorInfo],  # param -> full name
                output_tensors: Dict[str, TensorInfo], # param -> full name
                *args, **kwargs) -> None:
        self.name = name
        self.op_type = op_type
        self.input_tensors = input_tensors 
        self.output_tensors = output_tensors
        self.debug = kwargs.get('debug', False)

        self.num_core_range : Container
        self.num_core_range = None

        self.final_intra_sbp_sigs : Dict[str, SbpSignature]  # tensor local name -> sbp_signature
        self.final_intra_sbp_sigs = {}
        self.final_inter_sbp_sigs : Dict[str, SbpSignature]
        self.final_inter_sbp_sigs = {}
        self._candidate_intra_sbp_sigs: List[Dict[str, SbpSignature]]  # tensor local name -> sbp_signature
        self._candidate_intra_sbp_sigs = []

    # estimate various cost

    def estimate_cost(self, intra_sbp_sigs: Dict[str, SbpSignature], inter_sbp_sigs: Dict[str, SbpSignature],
                       arch_config: ArchConfig, training_config: TrainingConfig, detailed_report=False) -> Union[float, Dict]:
        """Estimate total cost of sbp signatures.

            @param intra_sbp_sigs: sbp internally seen by computation
            @param inter_sbp_sigs: input/output sbp externally seen by boxing. 
        """
        if self.debug:
            logger.debug(f"Operator {self.name} is estimating cost for SBP signature:")
            logger.debug(f"intra sbp signatures: {intra_sbp_sigs}")
            logger.debug(f"inter sbp signatures: {inter_sbp_sigs}")
            
        transmission_cost = self.estimate_transmission_cost(intra_sbp_sigs, inter_sbp_sigs, arch_config)
        compute_cost = self.estimate_compute_cost(intra_sbp_sigs, arch_config)
        sram_cost = self.estimate_sram_cost(intra_sbp_sigs, inter_sbp_sigs, arch_config, training_config)
        
        if self.debug:
            logger.debug(f"Summary")
            logger.debug(f"Transmission cost : {int(transmission_cost):>20}")
            logger.debug(f"Compute cost      : {int(compute_cost):>20}")
            logger.debug(f"SRAM cost         : {(0 if sram_cost == 0 else 'INF'):>20}")

        report = {
            'transmission_cost': transmission_cost,
            'compute_cost': compute_cost,
            'sram_cost': sram_cost,
        }

        if detailed_report:
            return report
        else:
            return sum(report.values())
        
    def estimate_transmission_cost(self, intra_sbp_sigs: Dict[str, SbpSignature], inter_sbp_sigs: Dict[str, SbpSignature],
                                   arch_config: ArchConfig) -> float:
        """Estimate transmission latency.
        Transmission for both input and output cannot be overlapped with compute for quick impl.
        Inter/intra input sbp signature need to have the same placement (unless this is a boxing operator).
        """
        if self.debug:
            logger.debug(f"Estimating transmission cost: ")

        comm_input_cost = 0
        for local_name, tensor_info in self.input_tensors.items():
            cur_sbp_sig = intra_sbp_sigs[local_name]
            prev_sbp_sig = inter_sbp_sigs.get(local_name, None)
            comm_input_cost += calc_comm_cost_on_same_devices(tensor_info, prev_sbp_sig, cur_sbp_sig, arch_config)
            if self.debug:
                logger.debug(f"comm input cost = {comm_input_cost} for {prev_sbp_sig} -> {cur_sbp_sig}")

        comm_output_cost = 0
        derived_output_sbp_signatures = derive_output_sbp_signatures(intra_sbp_sigs, self._rule_table)
        for local_name, tensor_info in self.output_tensors.items():
            prev_sbp_sig = derived_output_sbp_signatures[local_name]
            cur_sbp_sig = inter_sbp_sigs.get(local_name, None)
            comm_output_cost += calc_comm_cost_on_same_devices(tensor_info, prev_sbp_sig, cur_sbp_sig, arch_config)
            if self.debug:
                logger.debug(f"comm output cost = {comm_output_cost} for {prev_sbp_sig} -> {cur_sbp_sig}")
        
        return comm_input_cost + comm_output_cost
    
    def estimate_compute_cost(self, intra_sbp_sigs: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        """Estimate computation latency
        We leave it to operators impl to consider SRAM bandwidth
        We ignore the effect of overlapping compute with transmission.
        """
        if self.debug:
            logger.debug(f"Estimating compute cost: ")
        fp_latency = self.get_fp_latency(intra_sbp_sigs, arch_config)
        bp_latency = self.get_bp_latency(intra_sbp_sigs, arch_config)
        if self.debug:
            logger.debug(f"FP latency: {int(fp_latency)} cycles")
            logger.debug(f"BP latency: {int(bp_latency)} cycles")
        return fp_latency + bp_latency

    def estimate_sram_cost(self, intra_sbp_sigs: Dict[str, SbpSignature], inter_sbp_sigs: Dict[str, SbpSignature], arch_config: ArchConfig, training_config: TrainingConfig) -> float:
        """Check if SRAM is enough
        """
        if self.debug:
            logger.debug(f"Estimating SRAM cost: ")
        available_sram = arch_config.get_sram_size()
        bp_sram_util = self.get_bp_dynamic_sram_utilization(intra_sbp_sigs, inter_sbp_sigs, training_config) + self.get_bp_dynamic_sram_utilization(intra_sbp_sigs, inter_sbp_sigs, training_config)
        # fp uses less sram than bp, thus is ignored
        # temp buffer for dynamic broadcasting is ignored
        if self.debug:
            logger.debug(f"bp_sram_util   : {int(bp_sram_util):>15d}")
            logger.debug(f"available sram : {available_sram:>15d}")
        if bp_sram_util <= available_sram:
            return 0
        else:
            return np.inf

    def generate_candidate_intra_sbp_sigs(self):
        assert self.num_core_range != None
        self._candidate_intra_sbp_sigs = []
        self._generate_candidate_intra_sbp_sigs()
        assert len(self._candidate_intra_sbp_sigs) > 0
    
    def find_best_sbp_signature(self, arch_config: ArchConfig, training_config:TrainingConfig,
                                inter_sbp_sigs: Dict[str, SbpSignature] = {},
                                ) -> Tuple[Dict[str, SbpSignature]]:
        """Find the best inter/intra sbp signatures.
        If inter_sbp_sig is not specified, automatically find one with minimum mem footprint
        """
        assert self._candidate_intra_sbp_sigs, "Derive candidate sbp signatures first!"
        best_cost = np.inf
        best_intra_sbp_sigs = None
        best_inter_sbp_sigs = None

        for idx, intra_sbp_sigs in enumerate(self._candidate_intra_sbp_sigs):
            inter_sbp_sigs_ = deepcopy(inter_sbp_sigs)
            inter_sbp_sigs_.update({name: inter_sbp_sigs.get(name, self.find_best_input_inter_sbp_sig(name, intra_sbp_sigs[name])) for name in self.input_tensors})
            inter_sbp_sigs_.update({name: inter_sbp_sigs.get(name, self.find_best_output_inter_sbp_sig(name, intra_sbp_sigs[name])) for name in self.output_tensors})
            cost = self.estimate_cost(intra_sbp_sigs, inter_sbp_sigs_, arch_config, training_config)

            # if cost != np.inf:
            #     if self.debug:
            #         logger.debug(f"Cost = {int(cost):>10d}, for {idx}-th candidate")
            if cost < best_cost:
                best_cost = cost
                best_intra_sbp_sigs = intra_sbp_sigs
                best_inter_sbp_sigs = inter_sbp_sigs_

        if best_intra_sbp_sigs is None: 
            raise RuntimeError(f"Cannot find valid sbp signature!")
        if self.debug:
            logger.debug(f"Best intra sbp signatures: {intra_sbp_sigs}")
            logger.debug(f"Best inter sbp signatures: {inter_sbp_sigs_}")
            logger.debug(f"Best Cost: {int(best_cost):>10d}")

        for t in chain(self.input_tensors.keys(), self.output_tensors.keys()):
            if t not in best_intra_sbp_sigs or t not in best_inter_sbp_sigs:
                raise ValueError(f"{t} not in SBP signatures")
        return best_intra_sbp_sigs, best_inter_sbp_sigs
    
    def find_best_input_inter_sbp_sig(self, tensor_name: str, intra_sbp_sig: SbpSignature) -> SbpSignature:
        """Find inter sbp signature that incurs minimum memory footprint
        """
        tensor = self.input_tensors[tensor_name]
        best_inter_sbp_sig = intra_sbp_sig
        best_local_tensor_size = get_local_tensor_info(tensor, intra_sbp_sig).size()

        broadcast_dims = [i for i, sbp_prl in enumerate(intra_sbp_sig.sbp_parallels) if sbp_prl.is_broadcast()]
        for broadcast_dims_ in chain([combinations(broadcast_dims, subset_size) for subset_size in range(1, len(broadcast_dims))]):
            # for each broadcast placement dim i, find a tensor dim d(i), such that the split is valid
            for pdim_2_tdim in combinations_with_replacement(range(len(tensor.shape)), len(broadcast_dims_)):
                inter_sbp_sig = deepcopy(intra_sbp_sig)
                inter_sbp_sig.sbp_parallels = [(SplitSbpParallel(pdim_2_tdim[pdim]) if pdim in broadcast_dims_ else sbp_prl) for pdim, sbp_prl in inter_sbp_sig.sbp_parallels]
                try:
                    local_tensor_size = get_local_tensor_info(tensor, inter_sbp_sig).size()
                except:
                    continue
                if local_tensor_size < best_local_tensor_size:
                    best_local_tensor_size = local_tensor_size
                    best_inter_sbp_sig = inter_sbp_sig
        return best_inter_sbp_sig
    
    def find_best_output_inter_sbp_sig(self, tensor_name: str, intra_sbp_sig: SbpSignature, reduce_partial=True) -> SbpSignature:
        """Find inter sbp signature that incurs minimum memory footprint
        """
        tensor = self.output_tensors[tensor_name]
        best_inter_sbp_sig = deepcopy(intra_sbp_sig)
        if reduce_partial:
            best_inter_sbp_sig.sbp_parallels = [sbp_prl if not sbp_prl.is_partial() else BroadcastSbpParallel() for sbp_prl in best_inter_sbp_sig.sbp_parallels]
        best_local_tensor_size = get_local_tensor_info(tensor, intra_sbp_sig).size()

        partial_dims = [i for i, sbp_prl in enumerate(intra_sbp_sig.sbp_parallels) if sbp_prl.is_partial()]
        for partial_dims_ in chain([combinations(partial_dims, subset_size) for subset_size in range(1, len(partial_dims))]):
            for pdim_2_tdim in combinations_with_replacement(range(tensor.shape), len(partial_dims_)):
                inter_sbp_sig = deepcopy(intra_sbp_sig)
                inter_sbp_sig.sbp_parallels = [(SplitSbpParallel(pdim_2_tdim[pdim]) if pdim in partial_dims_ else sbp_prl) for pdim, sbp_prl in inter_sbp_sig.sbp_parallels]
                try:
                    local_tensor_size = get_local_tensor_info(tensor, inter_sbp_sig).size()
                except:
                    continue
                if local_tensor_size < best_local_tensor_size:
                    best_local_tensor_size = local_tensor_size
                    if reduce_partial:
                        inter_sbp_sig.sbp_parallels = [sbp_prl if not sbp_prl.is_partial() else BroadcastSbpParallel() for sbp_prl in inter_sbp_sig.sbp_parallels]
                    best_inter_sbp_sig = inter_sbp_sig
        return best_inter_sbp_sig
    
    # methods for characteristics of computing a local tensor on single core

    def get_bp_static_sram_utilization(self, intra_sbp_sigs: Dict[str, SbpSignature], inter_sbp_sigs: Dict[str, SbpSignature], training_config: TrainingConfig) -> int:
        """During bp computation of current batch, these static sram cannot be released.
        We consider pipeline-parallelism, where activation of the micro-batch cannot be swapped out to DRAM.
        """
        sram_util = 0

        # only consider input activation of the operator
        # They can be released before finishing this bp stage, but that's too complicated to analyze
        for name, tensor in self.input_tensors.items():
            sbp = inter_sbp_sigs[name]
            local_tensor = get_local_tensor_info(tensor, sbp)
            if local_tensor.kind in ['activation', 'input']:
                sram_util += local_tensor.size()

        return sram_util

    def get_bp_dynamic_sram_utilization(self, intra_sbp_sigs: Dict[str, SbpSignature], inter_sbp_sigs: Dict[str, SbpSignature], training_config: TrainingConfig) -> int:
        """During bp computation of current batch, these temporary sram buffer need to be allocated and 
        can be immediately released after bp finishes.

        These includes:
        - output tensors' grad
        - weight tensor itself, its' grad, dynamic os, static os
        - input tensors' grad

        Naive estimation of upper bound:
        - output tensor grad lives throughtout this bp
        - a copy of original weight lives throughout this bp
        - update one weight at a time, and flush updated weight
        - calc all activation's weight
        """

        static_sram_util = 0
        for name, tensor in self.output_tensors.items():
            sbp = intra_sbp_sigs[name]
            local_tensor = get_local_tensor_info(tensor, sbp)
            if local_tensor.kind in ['activation']:
                output_grad_size = local_tensor.size()
                static_sram_util += output_grad_size

        for name, tensor in self.input_tensors.items():
            sbp = inter_sbp_sigs[name]
            local_tensor = get_local_tensor_info(tensor, sbp)
            if local_tensor.kind == 'weight':
                weight_size = local_tensor.size()
                static_sram_util += weight_size

        weight_dynamic_sram_util = 0
        for name, tensor in self.input_tensors.items():
            sbp = inter_sbp_sigs[name]
            local_tensor = get_local_tensor_info(tensor, sbp)
            if local_tensor.kind in ['weight']:
                weight_copy_size = weight_grad_size = local_tensor.size()
                weight_dynamic_os_size = training_config.get_dynamic_optimizer_state_size() * local_tensor.numel()
                weight_static_os_size = training_config.get_static_optimizer_state_size() * local_tensor.numel()
                weight_dynamic_sram_util = max(weight_dynamic_sram_util, weight_copy_size + weight_grad_size + weight_dynamic_os_size + weight_static_os_size)

        input_dynamic_sram_util = 0
        for name, tensor in self.input_tensors.items():
            sbp = inter_sbp_sigs[name]
            local_tensor = get_local_tensor_info(tensor, sbp)

            if local_tensor.kind in ['input', 'activation']:
                input_grad_size = local_tensor.size()
                input_dynamic_sram_util += input_grad_size

        dynamic_sram_util = max(weight_dynamic_sram_util, input_dynamic_sram_util)
        total_sram_util = static_sram_util + dynamic_sram_util

        return total_sram_util
    
    @abstractmethod
    def get_fp_latency(self, intra_sbp_sigs: Dict[str, SbpSignature], arch_config: ArchConfig):
        raise NotImplementedError
    
    @abstractmethod
    def get_bp_latency(self, intra_sbp_sigs: Dict[str, SbpSignature], arch_config: ArchConfig):
        raise NotImplementedError
    
    @abstractmethod
    def get_fp_mac_count(self):
        raise NotImplementedError

    @abstractmethod
    def get_bp_mac_count(self):
        raise NotImplementedError

    def _estimate_latency_with_roofline_model(self, mac_count, sram_access_count, arch_config: ArchConfig):
        arithmetic_intensity = mac_count / sram_access_count

        compute_power = arch_config.get_compute_power()          # mac / cycle
        memory_bandwidth = arch_config.get_sram_bandwidth()    # byte / cycle
        maximum_intensity = compute_power / memory_bandwidth     # mac / byte

        if arithmetic_intensity < maximum_intensity:
            available_compute_power = memory_bandwidth * arithmetic_intensity  # memory bounded
        else:
            available_compute_power = compute_power  # compute bounded
        total_cycles = mac_count / available_compute_power

        # if self.debug:
        #     logger.debug(f"arithmetic intensity: {arithmetic_intensity}")
        #     logger.debug(f"maximum intensity: {maximum_intensity}")
        #     logger.debug(f"available compute power: {available_compute_power}")
        return total_cycles

    # sbp derivation

    @abstractmethod
    def _generate_candidate_intra_sbp_sigs(self) -> None:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def _rule_table(self) -> pd.DataFrame:
        """ columns = tensor local name, index = rule number
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def _dim_table(self) -> pd.DataFrame:
        """ columns = tensor local name, index = rule number
        """