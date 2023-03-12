
import os
import sys
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Container
from itertools import chain

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import (
    ArchConfig, SbpSignature, TensorInfo, calc_comm_cost_for_input, calc_comm_cost_for_reduction,
    derive_output_sbp_signature, derive_reduced_sbp_signatures,
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
        for param_name, current_sbp_signature in sbp_signatures.items():
            tensor_info = self.input_tensors[param_name]
            previous_sbp_signatures = inter_layer_sbp_signatures[tensor_info.name]
            comm_input_cost += calc_comm_cost_for_input(previous_sbp_signatures, current_sbp_signature, 
                                                        arch_config=arch_config, tensor_info=tensor_info)

        compute_cost = self._estimate_compute_cost(self, sbp_signatures, arch_config)
        memory_cost = self._estimate_memmory_cost(self, sbp_signatures, arch_config)

        input_sbp_signatures = [sbp for name, sbp in sbp_signatures.items if name in self.input_tensors]
        output_sbp_signatures = derive_output_sbp_signature(input_sbp_signatures, self._rule_table)

        comm_reduce_cost = 0
        for param_name, previous_sbp_signature in output_sbp_signatures.items():
            tensor_info = self.output_tensors[param_name]
            current_sbp_signature = sbp_signatures[param_name]
            comm_reduce_cost += calc_comm_cost_for_reduction(previous_sbp_signature, current_sbp_signature, 
                                                             arch_config=arch_config, tensor_info=tensor_info)
        
        return comm_input_cost + compute_cost + comm_reduce_cost + memory_cost

    def generate_candidate_sbp_signatures(self):
        assert self.num_core_range != None
        self._candidate_sbp_signatures = []
        self._generate_candidate_sbp_signatures()
        assert len(self._candidate_sbp_signatures) > 0
    
    def find_best_sbp_signature(self, arch_config: ArchConfig, inter_layer_sbp_signatures: Dict[str, SbpSignature] = {}
                                ) -> Dict[str, SbpSignature]:
        assert self._candidate_sbp_signatures, "Derive candidate sbp signatures first!"

        costs = [self.estimate_cost(sbp, arch_config, inter_layer_sbp_signatures)
                 for sbp in self._candidate_sbp_signatures]
        best_sbp_signatures = self._candidate_sbp_signatures[np.argmin(costs)]

        for t in chain(self.input_tensors.keys(), self.output_tensors.keys()):
            if t not in best_sbp_signatures:
                raise ValueError(f"{t} not in SBP signatures")
        return best_sbp_signatures
    
    @abstractmethod
    def _estimate_compute_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        """We use roofline model to estimate actual computation cost.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _estimate_memory_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig):
        """0 for enough memory, np.inf for invlaid memory
        """
        raise NotImplementedError
    
    @abstractmethod
    def _generate_candidate_sbp_signature():
        raise NotImplementedError
    
    @abstractmethod
    @property
    def _rule_table(self) -> pd.DataFrame:
        raise NotImplementedError