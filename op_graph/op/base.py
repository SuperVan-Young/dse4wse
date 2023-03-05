
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from op_graph import SbpSignature
from itertools import chain

# Estimation level of operator's latency
ESTIMATE_COMM = 1
ESTIMATE_COMP_COMM = 2

class Operator(ABC):

    def __init__(self, 
                name: str, 
                op_type: str, 
                input_tensors: Dict[str, str],
                output_tensors: Dict[str, str],
                shapes: Dict[str, Tuple[int]],
                *args, **kwargs) -> None:
        self.name = name
        self.op_type = op_type
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.shapes = shapes

        self.sbp_signatures : Dict[str, SbpSignature]
        self.sbp_signatures = {}

    def estimate_cost(self, level, inter_layer_sbp: Dict[str, SbpSignature]) -> float:
        cost = 0
        if level == ESTIMATE_COMM:
            cost += self._estimate_communication_cost(inter_layer_sbp)
        elif level == ESTIMATE_COMP_COMM:
            cost += self._estimate_communication_cost(inter_layer_sbp)
            cost += self._estimate_computation_cost()
        else:
            raise NotImplementedError(f"Estimation level {level} not implemented")
        return cost
    
    @abstractmethod
    def _estimate_communication_cost(self, inter_layer_sbp: Dict[str, SbpSignature]) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _estimate_computation_cost(self) -> float:
        raise NotImplementedError
    
    def derive_output_shapes(self) -> None:
        for inp in self.input_tensors.values():
            assert inp in self.shapes, f"Missing shape of input tensor {inp}"
        self._derive_output_shapes()
        for oup in self.output_tensors.values():
            assert inp in self.shapes, f"Missing shape of output tensor {oup}"

    @abstractmethod
    def _derive_output_shapes(self) -> None:
        raise NotImplementedError
    
    def find_best_sbp_signature(self, 
                                max_devices: int, 
                                inter_layer_sbp: Dict[str, SbpSignature] = {}
                                ) -> Dict[str, SbpSignature]:
        sbp = self._find_best_sbp_signature(max_devices, inter_layer_sbp)
        for t in chain(self.input_tensors.values(), self.output_tensors.values()):
            if t not in sbp:
                raise ValueError(f"{t} not in SBP signatures")
        return sbp
    
    @abstractmethod
    def _find_best_sbp_signature(self,
                                max_devices: int, 
                                inter_layer_sbp: Dict[str, SbpSignature] = {}
                                ) -> Dict[str, SbpSignature]:
        raise NotImplementedError