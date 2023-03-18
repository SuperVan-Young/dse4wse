
import os
import sys
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Container, Union
from itertools import chain

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import (
    ArchConfig, SbpSignature, TensorInfo, calc_comm_cost_for_input, calc_comm_cost_for_reduction,
    derive_output_sbp_signature, derive_reduced_sbp_signatures, logger, get_split_tensor_info
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

        self.num_core_range : Container
        self.num_core_range = None
        self._candidate_sbp_signatures: List[Dict[str, SbpSignature]]  # param name -> sbp_signature
        self._candidate_sbp_signatures = []
        self.final_sbp_signatures : Dict[str, SbpSignature]  # param name -> sbp_signature
        self.final_sbp_signatures = {}

    def estimate_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig, 
                      inter_layer_sbp_signatures: Dict[str, SbpSignature]) -> float:
        comm_input_cost = 0
        for param_name, tensor_info in self.input_tensors.items():
            current_sbp_signature = sbp_signatures[param_name]
            if not tensor_info.inplace:
                previous_sbp_signatures = inter_layer_sbp_signatures.get(tensor_info.name, None)
                comm_input_cost += calc_comm_cost_for_input(previous_sbp_signatures, current_sbp_signature, 
                                                            arch_config=arch_config, tensor_info=tensor_info)

        compute_cost = self.estimate_compute_cost(sbp_signatures, arch_config)

        memory_cost = self.estimate_memory_cost(sbp_signatures, arch_config)

        comm_reduce_cost = 0
        input_sbp_signatures = {name: sbp for name, sbp in sbp_signatures.items() if name in self.input_tensors}
        output_sbp_signatures = derive_output_sbp_signature(input_sbp_signatures, self._rule_table)
        for param_name, tensor_info in self.output_tensors.items():
            previous_sbp_signature = output_sbp_signatures[param_name]
            current_sbp_signature = sbp_signatures[param_name]
            comm_reduce_cost += calc_comm_cost_for_reduction(previous_sbp_signature, current_sbp_signature, 
                                                             arch_config=arch_config, tensor_info=tensor_info)
        
        # logger.debug(f"Estimating cost for SBP signature {sbp_signatures}")
        # logger.debug(f"input comm cost : {comm_input_cost:>20}")
        # logger.debug(f"Compute cost    : {compute_cost:>20}")
        # logger.debug(f"Reduce comm cost: {comm_reduce_cost:>20}")
        # logger.debug(f"Memory cost     : {memory_cost:>20}")

        return comm_input_cost + compute_cost + comm_reduce_cost + memory_cost
    
    def estimate_compute_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        """Estimate execution latency with roofline model.
        """
        input_tensors = {name: get_split_tensor_info(tensor_info, sbp_signatures[name])
                         for name, tensor_info in self.input_tensors.items()}
        mac_count = self.get_mac_count(input_tensors)            # mac
        mem_ref_count = self.get_mem_ref_count(input_tensors)    # byte
        operational_intensity = mac_count / mem_ref_count        # mac / byte

        compute_power = arch_config.get_compute_power()          # mac / cycle
        memory_bandwidth = arch_config.get_memory_bandwidth()    # byte / cycle
        maximum_intensity = compute_power / memory_bandwidth     # mac / byte

        if operational_intensity < maximum_intensity:
            available_compute_power = memory_bandwidth * operational_intensity
        else:
            available_compute_power = compute_power

        total_cycles = mac_count / available_compute_power
        return total_cycles

    def estimate_memory_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        input_tensors = {name: get_split_tensor_info(tensor_info, sbp_signatures[name])
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

        logger.debug(f"Candidate sbp signatures for {self.__class__}: {self.name}")
        for idx, sbp in enumerate(self._candidate_sbp_signatures):
            cost = self.estimate_cost(sbp, arch_config, inter_layer_sbp_signatures)
            if cost != np.inf:
                logger.debug(f"Cost = {int(cost):>10d}, for {idx}-th {sbp}")
            if cost < best_cost:
                best_cost = cost
                best_sbp_signatures = sbp 

        if best_sbp_signatures is None: 
            raise RuntimeError(f"Cannot find valid sbp signature!")
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
        raise NotImplementedError