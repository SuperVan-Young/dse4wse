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
    ArchConfig, SbpSignature, TensorInfo, factoring, Placement, SplitSbpParallel, BroadcastSbpParallel
)

class BinaryElementwiseOperator(Operator):
    def __init__(self, name: str, op_type: str, input_tensors: Dict[str, TensorInfo], output_tensors: Dict[str, TensorInfo],
                 operation_intensity=1, *args, **kwargs) -> None:
        super().__init__(name, op_type, input_tensors, output_tensors, *args, **kwargs)
        self.operation_intensity = operation_intensity
        assert len(input_tensors) == 2
        assert len(output_tensors) == 1
        assert 'A' in input_tensors
        assert 'B' in input_tensors
        assert 'out' in output_tensors

    def _estimate_compute_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig) -> float:
        out_info = self.output_tensors['out']
        output_sbp_signature = sbp_signatures['out']
        assert output_sbp_signature.get_partial_size() == 1
        output_numel = out_info.numel() * output_sbp_signature.get_broadcast_size() // output_sbp_signature.get_split_size()
        total_operation = output_numel * self.operation_intensity

        compute_power = arch_config.get_compute_power()  # operation/cycle
        memory_bandwidth = arch_config.get_memory_bandwidth() // out_info.dtype_size  # element/cycle
        maximum_intensity = compute_power / memory_bandwidth  # operation/element

        if self.operation_intensity < maximum_intensity:
            actual_intensity = memory_bandwidth * self.operation_intensity
        else:
            actual_intensity = compute_power
        total_cycles = total_operation / actual_intensity
        return total_cycles

    def _estimate_memory_cost(self, sbp_signatures: Dict[str, SbpSignature], arch_config: ArchConfig):
        A_info, B_info, out_info = self.input_tensors['A'], self.input_tensors['B'], self.output_tensors['out']
        A_sbp_sig, B_sbp_sig, out_sbp_sig = sbp_signatures['A'], sbp_signatures['B'], sbp_signatures['out']
        used_memory = A_info.numel() * A_info.dtype_size / A_sbp_sig.get_split_size() \
                    + B_info.numel() * B_info.dtype_size / B_sbp_sig.get_split_size() \
                    + out_info.numel() * out_info.dtype_size / out_sbp_sig.get_split_size()
        actual_memory = arch_config.get_memory_size()

        return 0 if used_memory < actual_memory else np.inf

    def _generate_candidate_sbp_signatures(self):
        A_info, B_info, out_info = self.input_tensors['A'], self.input_tensors['B'], self.output_tensors['out']
        candidate_sbp_signatures = []

        A_shape_dims, B_shape_dims, out_shape_dims = len(A_info.shape), len(B_info.shape), len(out_info.shape)
        A_offset, B_offset = out_shape_dims - A_shape_dims, out_shape_dims - B_shape_dims

        for array_dim in [1, 2]:
            for dims in combinations(list(range(out_shape_dims)), array_dim):
                out_dim_values = [out_info.shape[dim] for dim in dims]
                out_dim_value_factors = [factoring(val) for val in out_dim_values]
                def validate_split(split):
                    total_split = reduce(lambda x, y: x * y, split)
                    return total_split in self.num_core_range
                possible_splits = [split for split in product(*out_dim_value_factors) if validate_split(split)]
                max_split = max(possible_splits, key=lambda s: reduce(lambda x, y: x * y, s))

                if 1 in max_split:
                    continue

                placement = Placement(shape=max_split)
                out_sbp_sig = SbpSignature(
                    placement,
                    [SplitSbpParallel(dim) for dim in dims],
                )
                def get_input_sbp_parallel(tensor_info, dim, offset):
                    dim_ = dim - offset
                    if dim_ >= 0 and tensor_info.shape[dim_] != 1:
                        return SplitSbpParallel(dim_)
                    else:
                        return BroadcastSbpParallel()

                A_sbp_sig = SbpSignature(
                    placement,
                    [get_input_sbp_parallel(A_info, dim, A_offset) for dim in dims]
                )
                B_sbp_sig = SbpSignature(
                    placement,
                    [get_input_sbp_parallel(B_info, dim, B_offset) for dim in dims]
                )
                sbp_signatures = {
                    'A': A_sbp_sig,
                    'B': B_sbp_sig,
                    'out': out_sbp_sig,
                }
                candidate_sbp_signatures.append(sbp_signatures)

        self._candidate_sbp_signatures = candidate_sbp_signatures


    @property
    def _rule_table(self) -> pd.DataFrame:
        A_info, B_info, out_info = self.input_tensors['A'], self.input_tensors['B'], self.output_tensors['out']
        A_shape_dims, B_shape_dims, out_shape_dims = len(A_info.shape), len(B_info.shape), len(out_info.shape)
        A_offset, B_offset = out_shape_dims - A_shape_dims, out_shape_dims - B_shape_dims

        data = {
            'A': ['B'],
            'B': ['B'],
            'out': ['B'],
        }
        def insert_rule(a, b, out):
            data['A'].append(a)
            data['B'].append(b)
            data['out'].append(out)

        for dim in range(out_shape_dims):
            A_dim, B_dim = dim - A_offset, dim - B_offset
            A_can_split = (A_dim >= 0 and A_info.shape[A_dim] > 1)
            B_can_split = (B_dim >= 0 and B_info.shape[B_dim] > 1)
            if A_can_split and B_can_split:
                insert_rule(f"S({A_dim})", f"S({B_dim})", f"S({dim})")
            if A_can_split:
                insert_rule(f"S({A_dim})", "B", f"S({dim})")
            if B_can_split:
                insert_rule("B", f"S({B_dim})", f"S({dim})")
        
        return pd.DataFrame(data)
                
if __name__ == "__main__":
    input_tensors = {
        'A': TensorInfo(
            (32, 512, 1),
            1,
            'test_A'
        ),
        'B': TensorInfo(
            (512, 768),
            1,
            'test_B'
        ),
    }
    output_tensors = {
        'out': TensorInfo(
            (32, 512, 768),
            1,
            'test_output'
        )
    }
    binary_op = BinaryElementwiseOperator(
        'test_binary',
        'add',
        input_tensors=input_tensors,
        output_tensors=output_tensors
    )
    binary_op.num_core_range = list(range(1, 1025))
    binary_op.generate_candidate_sbp_signatures()

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
    best_sbp = binary_op.find_best_sbp_signature(arch_config=arch_config)