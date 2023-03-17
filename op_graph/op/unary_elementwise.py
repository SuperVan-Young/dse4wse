import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
from itertools import combinations, product
from functools import reduce

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base import Operator
from utils import (
    ArchConfig, Placement, SplitSbpParallel, SbpSignature, TensorInfo, factoring
)


class UnaryElementwiseOperator(Operator):
    def __init__(self, name: str, op_type: str, input_tensors: Dict[str, TensorInfo], output_tensors: Dict[str, TensorInfo], 
                 operation_intensity=1, *args, **kwargs) -> None:
        super().__init__(name, op_type, input_tensors, output_tensors, *args, **kwargs)
        self.operation_intensity = operation_intensity  # operation / element
        assert len(input_tensors) == 1, input_tensors
        assert len(output_tensors) == 1, output_tensors
        assert "in" in input_tensors
        assert "out" in output_tensors

    def _estimate_compute_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        tensor_info = self.input_tensors['in']
        sbp_signature = sbp_signatures['in']
        assert sbp_signature.get_partial_size() == 1
        input_numel = tensor_info.numel() * sbp_signature.get_broadcast_size() // sbp_signature.get_split_size()
        total_operation = input_numel * self.operation_intensity

        compute_power = arch_config.get_compute_power()  # operation/cycle
        memory_bandwidth = arch_config.get_memory_bandwidth() // tensor_info.dtype_size()  # element/cycle
        maximum_intensity = compute_power / memory_bandwidth  # operation/element

        if self.operation_intensity < maximum_intensity:
            actual_intensity = memory_bandwidth * self.operation_intensity
        else:
            actual_intensity = compute_power
        total_cycles = total_operation / actual_intensity
        return total_cycles
    
    def _estimate_memory_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig):
        tensor_info = self.input_tensors['in']
        sbp_signature = sbp_signatures['in']
        used_memory = tensor_info.numel() // sbp_signature.get_split_size() * tensor_info.dtype_size() * 2 # input + output
        actual_memory = arch_config.get_memory_size()

        return 0 if used_memory < actual_memory else np.inf
    
    def _generate_candidate_sbp_signatures(self):
        tensor_info = self.input_tensors['in']
        candidate_sbp_signatures = []

        for array_dim in [1, 2]:
            for dims in combinations(list(range(len(tensor_info.shape))), array_dim):
                dim_values = [tensor_info.shape[dim] for dim in dims]
                dim_value_factors = [factoring(val) for val in dim_values]
                def validate_split(split):
                    total_split = reduce(lambda x, y: x * y, split)
                    return total_split in self.num_core_range
                possible_splits = [split for split in product(*dim_value_factors) if validate_split(split)]
                max_split = max(possible_splits, key=lambda s: reduce(lambda x, y: x * y, s))
                sbp_signature = SbpSignature(
                    Placement(shape=max_split), 
                    [SplitSbpParallel(dim) for dim in dims]
                )
                candidate_sbp_signatures.append(sbp_signature)
        self._candidate_sbp_signatures = candidate_sbp_signatures
    
    @property
    def _rule_table(self) -> pd.DataFrame:
        tensor_info = self.input_tensors['in']
        num_dims = len(tensor_info.shape)
        data = {
            'in': ['B'] + [f"S({i})" for i in range(num_dims)],
            'out': ['B'] + [f"S({i})" for i in range(num_dims)],
        }
        return pd.DataFrame(data)
    
if __name__ == "__main__":
    input_tensors = {
        'in': TensorInfo(
            (32, 512, 768),
            1,
            'test_input'
        )
    }
    output_tensors = {
        'out': TensorInfo(
            (32, 512, 768),
            1,
            'test_output'
        )
    }
    unary_op = UnaryElementwiseOperator(
        'test_unary',
        'log',
        input_tensors=input_tensors,
        output_tensors=output_tensors
    )
    unary_op.num_core_range = list(range(1, 1025))
    unary_op.generate_candidate_sbp_signatures()
    print(unary_op._candidate_sbp_signatures)