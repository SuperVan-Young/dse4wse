
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
    derive_output_sbp_signatures, logger, get_local_tensor_info
)

class Operator(ABC):

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

    def estimate_cost(self, sbp_signatures: Dict[str, SbpSignature], input_sbp_signatures: Dict[str, SbpSignature],
                       arch_config: ArchConfig, detailed_report=False) -> Union[float, Dict]:
        """Estimate total cost of sbp signatures.

            @param sbp_signatures: input sbps demanded, and output sbps guaranteed by current operator
            @param input_sbp_signatures: actual input sbps, param -> sbp_signature
            @param arch_config: architecture description
        """
        transmission_cost = self.estimate_transmission_cost(sbp_signatures, input_sbp_signatures, arch_config)
        compute_cost = self.estimate_compute_cost(sbp_signatures, arch_config)
        memory_cost = self.estimate_memory_cost(sbp_signatures, arch_config)
        
        if self.is_debug:
            logger.debug(f"Estimating cost for SBP signature {sbp_signatures}")
            logger.debug(f"Transmission cost : {int(transmission_cost):>20}")
            logger.debug(f"Compute cost      : {int(compute_cost):>20}")
            logger.debug(f"Memory cost       : {(0 if memory_cost == 0 else 'INF'):>20}")

        report = {
            'transmission_cost': transmission_cost,
            'compute_cost': compute_cost,
            'memory_cost': memory_cost,
        }

        if detailed_report:
            return report
        else:
            return sum(report.values())
        
    def estimate_transmission_cost(self, sbp_signatures: Dict[str, SbpSignature], input_sbp_signatures: Dict[str, SbpSignature],
                                   arch_config: ArchConfig) -> float:
        """Estimate transmission latency.
        Transmission for both input and output cannot be overlapped with compute for quick impl.
        Input tensors are required to be on the same devices (unless this is a boxing operator).
        """
        comm_input_cost = 0
        for local_name, tensor_info in self.input_tensors.items():
            cur_sbp_sig = sbp_signatures[local_name]
            prev_sbp_sig = input_sbp_signatures.get(local_name, None)
            comm_input_cost += calc_comm_cost_on_same_devices(tensor_info, prev_sbp_sig, cur_sbp_sig, arch_config)

        comm_output_cost = 0
        derived_output_sbp_signatures = derive_output_sbp_signatures(input_sbp_signatures, self._rule_table)
        for local_name, tensor_info in self.output_tensors.items():
            prev_sbp_sig = derived_output_sbp_signatures[local_name]
            cur_sbp_sig = sbp_signatures[local_name]
            comm_output_cost += calc_comm_cost_on_same_devices(tensor_info, prev_sbp_sig, cur_sbp_sig, arch_config)
        
        return comm_input_cost + comm_output_cost
    
    def estimate_compute_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        """Estimate execution latency with roofline model.
        Input comm and compute are decoupled for quick impl.
        Virtual transform is essentially overlapping comm with input, but this effect is ignored.
        """
        input_tensors = {name: get_local_tensor_info(tensor_info, sbp_signatures[name])
                         for name, tensor_info in self.input_tensors.items()}
        mac_count = self.get_mac_count(input_tensors)            # mac
        mem_ref_count = self.get_mem_ref_count(input_tensors)    # byte
        operational_intensity = mac_count / mem_ref_count        # mac / byte

        compute_power = arch_config.get_compute_power()          # mac / cycle
        memory_bandwidth = arch_config.get_memory_bandwidth()    # byte / cycle
        maximum_intensity = compute_power / memory_bandwidth     # mac / byte

        if operational_intensity < maximum_intensity:
            available_compute_power = memory_bandwidth * operational_intensity  # memory bounded
        else:
            available_compute_power = compute_power  # compute bounded

        if self.is_debug:
            logger.debug(f"MAC count: {mac_count}")
            logger.debug(f"operational intensity: {operational_intensity}")
            logger.debug(f"maximum intensity: {maximum_intensity}")

        total_cycles = mac_count / available_compute_power
        return total_cycles

    def estimate_memory_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        """Estimate memory cost.
        We only consider memory held up by the operator's output tensor, i.e. its product;
        The runtime overhead of virtual transformation, i.e. the temporary buffer size, is ignored.
        Manipulating the original input tensors without making a deepcopy is technically wrong, 
        but we ignore the overhead of making another copy.
        """
        input_tensors = {name: get_local_tensor_info(tensor_info, sbp_signatures[name])
                         for name, tensor_info in self.input_tensors.items()}
        mem_utilization = self.get_mem_utilization(input_tensors)
        available_memory = arch_config.get_memory_size()

        return 0 if mem_utilization <= available_memory else np.inf

    def generate_candidate_sbp_signatures(self):
        assert self.num_core_range != None
        self._candidate_sbp_signatures = []
        self._generate_candidate_sbp_signatures()
        assert len(self._candidate_sbp_signatures) > 0
    
    def find_best_sbp_signature(self, arch_config: ArchConfig, inter_layer_sbp_signatures: Dict[str, SbpSignature] = {}
                                ) -> Dict[str, SbpSignature]:
        assert self._candidate_sbp_signatures, "Derive candidate sbp signatures first!"
        best_cost = np.inf
        best_sbp_signatures = None

        for idx, sbp in enumerate(self._candidate_sbp_signatures):
            cost = self.estimate_cost(sbp, arch_config, inter_layer_sbp_signatures)
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
    
    def get_mac_count(self, input_tensors: Union[None, Dict[str, TensorInfo]] = None):
        if input_tensors is None:
            input_tensors = self.input_tensors
        return self._get_mac_count(input_tensors)
    
    def get_mem_ref_count(self, input_tensors: Union[None, Dict[str, TensorInfo]] = None):
        if input_tensors is None:
            input_tensors = self.input_tensors
        return self._get_mem_ref_count(input_tensors)
    
    def get_mem_utilization(self, input_tensors: Union[None, Dict[str, TensorInfo]] = None, **kwargs):
        # TODO: in the future, consider training memory utilization
        if input_tensors is None:
            input_tensors = self.input_tensors
        return self._get_mem_utilization(input_tensors)
    
    @abstractmethod
    def _get_mac_count(self, input_tensors: Dict[str, TensorInfo]):
        raise NotImplementedError
    
    @abstractmethod
    def _get_mem_ref_count(self, input_tensors: Dict[str, TensorInfo]):
        raise NotImplementedError
    
    @abstractmethod
    def _get_mem_utilization(self, input_tensors: Dict[str, TensorInfo]):
        raise NotImplementedError
    
    @abstractmethod
    def _generate_candidate_sbp_signatures(self) -> None:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def _rule_table(self) -> pd.DataFrame:
        """ columns = tensor local name, index = rule number
        """
        raise NotImplementedError