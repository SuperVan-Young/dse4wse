
import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
from itertools import combinations, product
from functools import reduce

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base import Operator
from utils import (
    ArchConfig, SbpSignature, TensorInfo, factoring, Placement, SplitSbpParallel, 
    BroadcastSbpParallel, multidirectional_broadcasting, logger
)

class MatMulOperator(Operator):
    def __init__(self, name: str, op_type: str, input_tensors: Dict[str, TensorInfo], output_tensors: Dict[str, TensorInfo],
                 M_block_size=1, K_block_size=1, N_block_size=1,  *args, **kwargs) -> None:
        super().__init__(name, op_type, input_tensors, output_tensors, *args, **kwargs)
        # assuming some lower-level memory hierarchy below SRAM, for comparing GPU and WSE
        self.M_block_size = M_block_size
        self.K_block_size = K_block_size
        self.N_block_size = N_block_size
        assert len(input_tensors) == 2
        assert len(output_tensors) == 1
        assert 'A' in input_tensors
        assert 'B' in input_tensors
        assert 'Y' in output_tensors
        assert len(input_tensors['A'].shape) > 1, "Currently not support VxM"
        assert len(input_tensors['B'].shape) > 1, "Currently not support MxV"

    def _get_mac_count(self, input_tensors: Dict[str, TensorInfo]):
        A_info, B_info = input_tensors['A'], input_tensors['B']
        stack_shape = multidirectional_broadcasting(A_info.shape[:-2], B_info.shape[:-2])
        M_dim_value, K_dim_value, N_dim_value = A_info.shape[-2], A_info.shape[-1], B_info.shape[-1]
        return reduce(lambda x, y: x * y, stack_shape + [M_dim_value, K_dim_value, N_dim_value])
    
    def _get_mem_ref_count(self, input_tensors: Dict[str, TensorInfo]):
        A_info, B_info = input_tensors['A'], input_tensors['B']
        stack_shape = multidirectional_broadcasting(A_info.shape[:-2], B_info.shape[:-2])
        M_dim_value, K_dim_value, N_dim_value = A_info.shape[-2], A_info.shape[-1], B_info.shape[-1]

        # assume no inner blocking
        A_ref_count = M_dim_value * K_dim_value * N_dim_value // self.N_block_size
        B_ref_count = M_dim_value * K_dim_value * N_dim_value // self.M_block_size
        Y_ref_count = M_dim_value * N_dim_value // self.K_block_size
        total_ref_count = reduce(lambda x, y: x * y, stack_shape, 1) * (A_ref_count + B_ref_count + Y_ref_count)

        return total_ref_count

    def _get_mem_utilization(self, input_tensors: Dict[str, TensorInfo]):
        # assume A, B, Y use separate memory
        A_info, B_info = input_tensors['A'], input_tensors['B']
        stack_shape = multidirectional_broadcasting(A_info.shape[:-2], B_info.shape[:-2])
        M_dim_value, N_dim_value = A_info.shape[-2], B_info.shape[-1]

        A_size = A_info.size()
        B_size = B_info.size()
        Y_size = reduce(lambda x, y: x * y, stack_shape + [M_dim_value, N_dim_value])
        
        return A_size + B_size + Y_size

    def _generate_candidate_sbp_signatures(self) -> None:
        candidate_sbp_signatures = []

        A_info, B_info, Y_info = self.input_tensors['A'], self.input_tensors['B'], self.output_tensors['Y']
        A_shape_dims, B_shape_dims, Y_shape_dims = len(A_info.shape), len(B_info.shape), len(Y_info.shape)
        A_offset, B_offset = Y_shape_dims - A_shape_dims, Y_shape_dims - B_shape_dims
        A_dim_M, A_dim_K = A_shape_dims - 2, A_shape_dims - 1
        B_dim_K, B_dim_N = B_shape_dims - 2, B_shape_dims - 1
        Y_dim_M, Y_dim_N = Y_shape_dims - 2, Y_shape_dims - 1
        M_dim_value, K_dim_value, N_dim_value = A_info.shape[-2], A_info.shape[-1], B_info.shape[-1]
        M_dim_value_factors, K_dim_value_factors, N_dim_value_factors = factoring(M_dim_value), factoring(K_dim_value), factoring(N_dim_value)

        def mul_reduce(s):
            return reduce(lambda x, y: x * y, s)

        def get_input_sbp_parallel(tensor_info, dim, offset):
            dim_ = dim - offset
            if dim_ >= 0 and tensor_info.shape[dim_] != 1:
                return SplitSbpParallel(dim_)
            else:
                return BroadcastSbpParallel()

        for num_stack_split_dim in range(0, min(Y_shape_dims - 2, 1) + 1):
            for stack_split_dims in combinations(list(range(Y_shape_dims - 2)), num_stack_split_dim):
                stack_split_dim_values = [Y_info.shape[dim] for dim in stack_split_dims]
                stack_split_dim_value_factors = [factoring(val) for val in stack_split_dim_values]
                
                possible_splits = [split for split in product(
                    M_dim_value_factors, N_dim_value_factors, K_dim_value_factors, *stack_split_dim_value_factors)
                    if mul_reduce(split) in self.num_core_range]
                candidate_splits = sorted(possible_splits, key=mul_reduce, reverse=True)
                # [:min(200, len(possible_splits))]
                # TODO: plot perf against utilized core

                for split in candidate_splits:
                    if 1 in split[3:]: continue  # covered in lower num_stack_split_dim

                    placement = Placement(shape=split)
                    Y_sbp_sig = SbpSignature(
                        placement,
                        [ SplitSbpParallel(Y_dim_M),
                          SplitSbpParallel(Y_dim_N),
                          BroadcastSbpParallel(),  # Partial -> Broadcast
                        ] + [SplitSbpParallel(dim) for dim in stack_split_dims]
                    )
                    A_sbp_sig = SbpSignature(
                        placement,
                        [ SplitSbpParallel(A_dim_M),
                          BroadcastSbpParallel(),
                          SplitSbpParallel(A_dim_K),
                        ] + [get_input_sbp_parallel(A_info, dim, A_offset) for dim in stack_split_dims]
                    )
                    B_sbp_sig = SbpSignature(
                        placement,
                        [ BroadcastSbpParallel(),
                          SplitSbpParallel(B_dim_N),
                          SplitSbpParallel(B_dim_K),
                        ] + [get_input_sbp_parallel(B_info, dim, B_offset) for dim in stack_split_dims]
                    )
                    sbp_signatures = {
                        'A': A_sbp_sig,
                        'B': B_sbp_sig,
                        'Y': Y_sbp_sig,
                    }
                    candidate_sbp_signatures.append(sbp_signatures)
        
        self._candidate_sbp_signatures = candidate_sbp_signatures
    
    @property
    def _rule_table(self) -> pd.DataFrame:
        A_info, B_info, Y_info = self.input_tensors['A'], self.input_tensors['B'], self.output_tensors['Y']
        A_shape_dims, B_shape_dims, Y_shape_dims = len(A_info.shape), len(B_info.shape), len(Y_info.shape)
        A_offset, B_offset = Y_shape_dims - A_shape_dims, Y_shape_dims - B_shape_dims
        A_dim_M, A_dim_K = A_shape_dims - 2, A_shape_dims - 1
        B_dim_K, B_dim_N = B_shape_dims - 2, B_shape_dims - 1
        Y_dim_M, Y_dim_N = Y_shape_dims - 2, Y_shape_dims - 1

        data = {
            'A': ['B'],
            'B': ['B'],
            'Y': ['B'],
        }
        def insert_rule(a, b, y):
            data['A'].append(a)
            data['B'].append(b)
            data['Y'].append(y)

        for dim in range(0, Y_shape_dims - 2):
            A_dim, B_dim = dim - A_offset, dim - B_offset
            A_can_split = (A_dim >= 0 and A_info.shape[A_dim] > 1)
            B_can_split = (B_dim >= 0 and B_info.shape[B_dim] > 1)
            if A_can_split and B_can_split:
                insert_rule(f"S({A_dim})", f"S({B_dim})", f"S({dim})")
            if A_can_split:
                insert_rule(f"S({A_dim})", "B", f"S({dim})")
            if B_can_split:
                insert_rule("B", f"S({B_dim})", f"S({dim})")

        insert_rule(f"S({A_dim_M})", "B", f"S({Y_dim_M})")
        insert_rule("B", f"S({B_dim_N})", f"S({Y_dim_N})")
        insert_rule(f"S({A_dim_K})", f"S({B_dim_K})", "P")

        return pd.DataFrame(data)
    
if __name__ == "__main__":
    input_tensors = {
        'A': TensorInfo(
            (1, 12, 512, 768),
            1,
            'test_A'
        ),
        'B': TensorInfo(
            (768, 768),
            1,
            'test_B'
        ),
    }
    output_tensors = {
        'Y': TensorInfo(
            (1, 12, 512, 768),
            1,
            'test_Y'
        )
    }
    matmul_op = MatMulOperator(
        'test_matmul',
        'MatMul',
        input_tensors=input_tensors,
        output_tensors=output_tensors
    )
    matmul_op.num_core_range = list(range(16384, 16384 + 1))
    matmul_op.generate_candidate_sbp_signatures()
    print(matmul_op._candidate_sbp_signatures)

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
    best_sbp = matmul_op.find_best_sbp_signature(arch_config=arch_config)