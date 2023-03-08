
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from itertools import chain

from op_graph import SbpSignature
from utils import ArchConfig

class Operator(ABC):

    def __init__(self, 
                name: str, 
                op_type: str, 
                input_tensors: Dict[str, str],  # param -> full name
                output_tensors: Dict[str, str], # param -> full name
                *args, **kwargs) -> None:
        self.name = name
        self.op_type = op_type
        self.input_tensors = input_tensors 
        self.output_tensors = output_tensors

        self.num_core_lower_bound = 0  # utilized_core in [lower_bound, upper_bound]
        self.num_core_upper_bound = 0
        self.final_sbp_signatures : Dict[str, SbpSignature]  # full name -> sbp_signature
        self.final_sbp_signatures = {}
        self._candidate_sbp_signatures: List[Dict[str, SbpSignature]]
        self._candidate_sbp_signatures = []

    def estimate_communication_cost(self, arch_config: ArchConfig, inter_layer_sbp: Dict[str, SbpSignature]) -> float:
        return self._estimate_communication_cost(arch_config, inter_layer_sbp)
    
    def estimate_computation_cost(self, arch_config: ArchConfig) -> float:
        return self._estimate_computation_cost(arch_config)
    
    def estimate_memory_cost(self) -> float:
        return self._estimate_memory_cost()
    
    def derive_output_shapes(self, input_shapes: Dict[str, Tuple[int]]) -> Dict[str, Tuple[int]]:
        for inp in self.input_tensors.values():
            assert inp in input_shapes, f"Missing shape of input tensor {inp}"
        return self._derive_output_shapes(input_shapes)
    
    def derive_candidate_sbp_signatures(self):
        assert self.num_core_lower_bound != 0
        assert self.num_core_upper_bound != 0
        self._candidate_sbp_signatures = []
        self._derive_candidate_sbp_signatures()
        assert len(self._candidate_sbp_signatures) > 0
    
    def find_best_sbp_signature(self, inter_layer_sbp: Dict[str, SbpSignature] = {}
                                ) -> Dict[str, SbpSignature]:
        assert self._candidate_sbp_signatures, "Derive candidate sbp signatures first!"
        sbp = self._find_best_sbp_signature(inter_layer_sbp)
        for t in chain(self.input_tensors.values(), self.output_tensors.values()):
            if t not in sbp:
                raise ValueError(f"{t} not in SBP signatures")
        return sbp
    
    @abstractmethod
    def _estimate_communication_cost(self, arch_config: ArchConfig, inter_layer_sbp: Dict[str, SbpSignature]) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _estimate_computation_cost(self, arch_config: ArchConfig) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _estimate_memory_cost(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def _derive_output_shapes(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _derive_candidate_sbp_signature():
        raise NotImplementedError
    
    @abstractmethod
    def _find_best_sbp_signature(self, inter_layer_sbp: Dict[str, SbpSignature] = {}
                                ) -> Dict[str, SbpSignature]:
        raise NotImplementedError