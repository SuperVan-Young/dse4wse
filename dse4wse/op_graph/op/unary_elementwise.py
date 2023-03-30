from typing import Dict, List
import numpy as np
import pandas as pd
from itertools import combinations, product
from functools import reduce

from .base import BaseOperator
from dse4wse.utils import (
    ArchConfig, Placement, SplitSbpParallel, BroadcastSbpParallel, SbpSignature, TensorInfo, factoring
)


class UnaryElementwiseOperator(BaseOperator):
    def __init__(self, name: str, op_type: str, input_tensors: Dict[str, TensorInfo], output_tensors: Dict[str, TensorInfo], 
                 mac_per_element=1, *args, **kwargs) -> None:
        super().__init__(name, op_type, input_tensors, output_tensors, *args, **kwargs)
        self.mac_per_element = mac_per_element  # operation / element
        assert len(input_tensors) == 1, input_tensors
        assert len(output_tensors) == 1, output_tensors
        assert "in" in input_tensors
        assert "out" in output_tensors
    
    def _get_mac_count(self, input_tensors: Dict[str, TensorInfo]):
        in_info = input_tensors['in']
        return in_info.numel() * self.mac_per_element
    
    def _get_mem_ref_count(self, input_tensors: Dict[str, TensorInfo]):
        in_info = input_tensors['in']
        in_size = in_info.size()
        return in_size * 2  # read in, write out

    def _get_mem_utilization(self, input_tensors: Dict[str, TensorInfo]):
        in_info = input_tensors['in']
        in_size = in_info.size()
        return in_size  # only save one copy!

    def _generate_candidate_sbp_signatures(self):
        tensor_info = self.input_tensors['in']
        candidate_sbp_signatures = []

        # Add default placement for single core scenarios
        if 1 in self.num_core_range:
            sbp_sig = SbpSignature(
                Placement(shape=[1], interconnect_types=['noc']),
                [BroadcastSbpParallel()],
            )
            sbp_sigs = {
                'in': sbp_sig,
                'out': sbp_sig,
            }
            candidate_sbp_signatures.append(sbp_sigs)

        for array_dim in [1, 2]:
            for dims in combinations(list(range(len(tensor_info.shape))), array_dim):
                dim_values = [tensor_info.shape[dim] for dim in dims]
                dim_value_factors = [factoring(val) for val in dim_values]
                def validate_split(split):
                    total_split = reduce(lambda x, y: x * y, split)
                    return total_split in self.num_core_range
                possible_splits = [split for split in product(*dim_value_factors) if validate_split(split)]
                if not possible_splits: continue  # e.g. split on dim_value = 1
                max_split = max(possible_splits, key=lambda s: reduce(lambda x, y: x * y, s))

                if 1 in max_split: continue  # covered by lower array_dim

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