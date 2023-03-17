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
        memory_bandwidth = arch_config.get_memory_bandwidth() // tensor_info.dtype_size  # element/cycle
        maximum_intensity = compute_power / memory_bandwidth  # operation/element

        if self.operation_intensity < maximum_intensity:
            actual_intensity = memory_bandwidth * self.operation_intensity
        else:
            actual_intensity = compute_power
        total_cycles = total_operation / actual_intensity
        return total_cycles
    
    def _estimate_memory_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig):
        in_info, out_info = self.input_tensors['in'], self.output_tensors['out']
        in_sbp_sig, out_sbp_sig = sbp_signatures['in'], sbp_signatures['out']
        used_memory = in_info.numel() * in_info.dtype_size * in_sbp_sig.get_broadcast_size() * in_sbp_sig.get_partial_size() / in_sbp_sig.get_split_size() \
                    + out_info.numel() * out_info.dtype_size * out_sbp_sig.get_broadcast_size() * out_sbp_sig.get_partial_size() / out_sbp_sig.get_split_size()
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

                if 1 in max_split:
                    continue

                sbp_signature = SbpSignature(
                    Placement(shape=max_split), 
                    [SplitSbpParallel(dim) for dim in dims]
                )
                sbp_signatures = {
                    'in': sbp_signature,
                    'out': sbp_signature,
                }
                candidate_sbp_signatures.append(sbp_signatures)
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

    arch_config = ArchConfig({
        'core_num_mac': 32,
        'core_buffer_width': 16,
        'core_buffer_size': 48 * 1024,
        'noc_virtual_channel': 4,
        'noc_buffer_size': 8,
        'noc_bandwidth': 4096,
        'core_array_height': 25,
        'core_array_width': 25,
        'reticle_array_height': 8,
        'reticle_array_width': 8,
        'inter_reticle_bandwidth': 1024,
        'inter_wafer_bandwidth': 256,
    })
    best_sbp = unary_op.find_best_sbp_signature(arch_config=arch_config)
    print(best_sbp)