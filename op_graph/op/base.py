
import os
import sys
import pandas as pd
import numpy as np
import math

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Container, Union
from itertools import chain

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import (
    ArchConfig, SbpSignature, TensorInfo, calc_comm_cost_on_same_devices,
    derive_output_sbp_signatures, logger, get_local_tensor_info, TrainingConfig
)

class Operator(ABC):
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
        self.is_debug = kwargs.get('debug', False)

        self.num_core_range : Container
        self.num_core_range = None

        self.final_sbp_signatures : Dict[str, SbpSignature]  # tensor local name -> sbp_signature
        self.final_sbp_signatures = {}
        self._candidate_sbp_signatures: List[Dict[str, SbpSignature]]  # tensor local name -> sbp_signature
        self._candidate_sbp_signatures = []

    # estimate various cost

    def estimate_cost(self, intra_sbp_sigs: Dict[str, SbpSignature], inter_sbp_sigs: Dict[str, SbpSignature],
                       arch_config: ArchConfig, training_config: TrainingConfig, detailed_report=False) -> Union[float, Dict]:
        """Estimate total cost of sbp signatures.

            @param intra_sbp_sigs: sbp signatures current operator uses, so that it doesn't need to consider inter op sbp sigs.
            @param inter_sbp_sigs: input sbps and output sbps
            @param arch_config: architecture description
        """
        if self.is_debug:
            logger.debug(f"Estimating cost for SBP signature {intra_sbp_sigs}")
            
        transmission_cost = self.estimate_transmission_cost(intra_sbp_sigs, inter_sbp_sigs, arch_config)
        compute_cost = self.estimate_compute_cost(intra_sbp_sigs, arch_config)
        sram_cost = self.estimate_sram_cost(intra_sbp_sigs, arch_config, training_config)
        
        if self.is_debug:
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
        Input tensors are required to be on the same devices (unless this is a boxing operator).
        """
        comm_input_cost = 0
        for local_name, tensor_info in self.input_tensors.items():
            cur_sbp_sig = intra_sbp_sigs[local_name]
            prev_sbp_sig = inter_sbp_sigs.get(local_name, None)
            comm_input_cost += calc_comm_cost_on_same_devices(tensor_info, prev_sbp_sig, cur_sbp_sig, arch_config)

        comm_output_cost = 0
        derived_output_sbp_signatures = derive_output_sbp_signatures(intra_sbp_sigs, self._rule_table)
        for local_name, tensor_info in self.output_tensors.items():
            prev_sbp_sig = derived_output_sbp_signatures[local_name]
            cur_sbp_sig = inter_sbp_sigs.get(local_name, None)
            comm_output_cost += calc_comm_cost_on_same_devices(tensor_info, prev_sbp_sig, cur_sbp_sig, arch_config)
        
        return comm_input_cost + comm_output_cost
    
    def estimate_compute_cost(self, intra_sbp_sigs: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        """Estimate computation latency
        We leave it to operators impl to consider SRAM bandwidth
        We ignore the effect of overlapping compute with transmission.
        """
        return self.get_fp_latency(intra_sbp_sigs, arch_config) + self.get_bp_latency(intra_sbp_sigs, arch_config)

    def estimate_sram_cost(self, intra_sbp_sigs: Dict[str, SbpSignature], arch_config: ArchConfig, training_config: TrainingConfig) -> float:
        """Check if SRAM is enough
        """
        available_sram = arch_config.get_sram_size()
        bp_sram_util = self.get_bp_dynamic_sram_utilization(intra_sbp_sigs, training_config) + self.get_bp_dynamic_sram_utilization(intra_sbp_sigs, training_config)
        # fp uses less sram than bp, thus is ignored
        # temp buffer for dynamic broadcasting is ignored
        if bp_sram_util <= available_sram:
            return 0
        else:
            if self.is_debug:
                logger.debug(f"bp_sram_util {bp_sram_util} > available sram {available_sram}")
            return np.inf

    def generate_candidate_sbp_signatures(self):
        assert self.num_core_range != None
        self._candidate_sbp_signatures = []
        self._generate_candidate_sbp_signatures()
        assert len(self._candidate_sbp_signatures) > 0
    
    def find_best_sbp_signature(self, arch_config: ArchConfig, training_config:TrainingConfig,
                                inter_sbp_sigs: Dict[str, SbpSignature] = {},
                                ) -> Dict[str, SbpSignature]:
        assert self._candidate_sbp_signatures, "Derive candidate sbp signatures first!"
        best_cost = np.inf
        best_sbp_signatures = None

        for idx, sbp in enumerate(self._candidate_sbp_signatures):
            cost = self.estimate_cost(sbp, inter_sbp_sigs, arch_config, training_config)
            if cost != np.inf:
                if self.is_debug:
                    logger.debug(f"Cost = {int(cost):>10d}, for {idx}-th {sbp}")
            if cost < best_cost:
                best_cost = cost
                best_sbp_signatures = sbp 

        if best_sbp_signatures is None: 
            raise RuntimeError(f"Cannot find valid sbp signature!")
        if self.is_debug:
            logger.debug(f"Best sbp signature: {best_sbp_signatures}")
            logger.debug(f"Best Cost: {int(best_cost):>10d}")

        for t in chain(self.input_tensors.keys(), self.output_tensors.keys()):
            if t not in best_sbp_signatures:
                raise ValueError(f"{t} not in SBP signatures")
        return best_sbp_signatures
    
    # methods for characteristics of computing a local tensor on single core

    def get_bp_static_sram_utilization(self, intra_sbp_sigs: Dict[str, SbpSignature], training_config: TrainingConfig) -> int:
        """During bp computation of current batch, these static sram cannot be released.
        We consider pipeline-parallelism, where activation of the micro-batch cannot be swapped out to DRAM.
        """
        sram_util = 0

        # only consider input activation of the operator
        # They can be released before finishing this bp stage, but that's too complicated to analyze
        for name, tensor in self.input_tensors.items():
            sbp = intra_sbp_sigs[name]
            local_tensor = get_local_tensor_info(tensor, sbp)
            if local_tensor.kind in ['activation', 'input']:
                sram_util += local_tensor.size()

        return sram_util

    def get_bp_dynamic_sram_utilization(self, intra_sbp_sigs: Dict[str, SbpSignature], training_config: TrainingConfig) -> int:
        """During bp computation of current batch, these temporary sram buffer need to be allocated and 
        can be immediately released after bp finishes.

        These includes:
        - output tensors' grad
        - weight tensor itself, its' grad, dynamic os, static os
        - input tensors' grad
        """
        sram_util = 0
        
        for name, tensor in self.input_tensors.items():
            sbp = intra_sbp_sigs[name]
            local_tensor = get_local_tensor_info(tensor, sbp)
            if local_tensor.kind in ['input', 'activation']:
                input_grad_size = local_tensor.size()
                sram_util += input_grad_size
            elif local_tensor.kind in ['weight']:
                weight_size = weight_grad_size = local_tensor.size()
                weight_dynamic_os_size = training_config.get_dynamic_optimizer_state_size() * local_tensor.numel()
                weight_static_os_size = training_config.get_static_optimizer_state_size() * local_tensor.numel()
                sram_util += weight_size + weight_grad_size + weight_dynamic_os_size + weight_static_os_size
            else:
                continue
        for name, tensor in self.output_tensors.items():
            sbp = intra_sbp_sigs[name]
            local_tensor = get_local_tensor_info(tensor, sbp)
            if local_tensor.kind in ['activation']:
                output_grad_size = local_tensor.size()
                sram_util += output_grad_size

        return sram_util
    
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
        return total_cycles

    # sbp derivation

    @abstractmethod
    def _generate_candidate_sbp_signatures(self) -> None:
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