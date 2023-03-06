
from typing import Dict, List, Tuple

from base import Operator
from op_graph import SbpSignature
from utils import ArchConfig

class EmptyOperator(Operator):

    def __init__(self, name: str, op_type: str, input_tensors: Dict[str, str], output_tensors: Dict[str, str], shapes: Dict[str, Tuple[int]], *args, **kwargs) -> None:
        super().__init__(name, op_type, input_tensors, output_tensors, shapes, *args, **kwargs)

    def _estimate_communication_cost(self, arch_config: ArchConfig, inter_layer_sbp: Dict[str, SbpSignature]) -> float:
        return 0
    
    def _estimate_computation_cost(self, arch_config: ArchConfig) -> float:
        return 0
    
    def _find_best_sbp_signature(self, max_devices: int, inter_layer_sbp: Dict[str, SbpSignature] = ...) -> Dict[str, SbpSignature]:
        # TODO: return default sbp signature
        raise NotImplementedError