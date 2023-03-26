
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
    ArchConfig, SbpSignature, TensorInfo, factoring, Placement, SplitSbpParallel, PartialSbpParallel,
    BroadcastSbpParallel, multidirectional_broadcasting, logger, transpose, get_local_tensor_info, TrainingConfig
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

    def get_fp_mac_count(self):
        A_info, B_info = self.input_tensors['A'], self.input_tensors['B']
        return self.__get_matmul_mac_count(A_info.shape, B_info.shape)
    
    def get_bp_mac_count(self):
        return self.__get_bp_A_mac_count() + self.__get_bp_B_mac_count()
    
    def __get_bp_A_mac_count(self):
        B_info, Y_info = self.input_tensors['B'], self.output_tensors['Y']
        shape_B_transpose = transpose(B_info.shape, -2, -1)
        return self.__get_matmul_mac_count(Y_info.shape, shape_B_transpose)
    
    def __get_bp_B_mac_count(self):
        A_info, Y_info = self.input_tensors['A'], self.output_tensors['Y']
        shape_A_transpose = transpose(A_info.shape, -2, -1)
        return self.__get_matmul_mac_count(shape_A_transpose, Y_info.shape)
    
    def __get_matmul_mac_count(self, shape0, shape1):
        stack_shape = multidirectional_broadcasting(shape0[:-2], shape1[:-2])
        M, K0, K1, N = shape0[-2], shape0[-1], shape1[-2], shape1[-1]
        assert K0 == K1, f"{K0} != {K1}"
        return reduce(lambda x, y: x * y, stack_shape + [M, K1, N])
    
    def __get_matmul_sram_access_count(self, shape0, shape1):
        stack_shape = multidirectional_broadcasting(shape0[:-2], shape1[:-2])
        M, K0, K1, N = shape0[-2], shape0[-1], shape1[-2], shape1[-1]
        assert K0 == K1

        # assume no inner blocking
        A_ref_count = M * K0 * N // self.N_block_size
        B_ref_count = M * K1 * N // self.M_block_size
        Y_ref_count = M * N // self.K_block_size
        total_ref_count = reduce(lambda x, y: x * y, stack_shape, 1) * (A_ref_count + B_ref_count + Y_ref_count)

        return total_ref_count
    
    def get_fp_latency(self, intra_sbp_sigs: Dict[str, SbpSignature], arch_config: ArchConfig):
        A_info, B_info = self.input_tensors['A'], self.input_tensors['B']
        A_local = get_local_tensor_info(A_info, intra_sbp_sigs['A'])
        B_local = get_local_tensor_info(B_info, intra_sbp_sigs['B'])
        mac_count = self.get_fp_mac_count()
        sram_access_count = self.__get_matmul_sram_access_count(A_local.shape, B_local.shape)
        return self._estimate_latency_with_roofline_model(mac_count, sram_access_count, arch_config)

    def get_bp_latency(self, intra_sbp_sigs: Dict[str, SbpSignature], arch_config: ArchConfig):
        A_info, B_info, Y_info = self.input_tensors['A'], self.input_tensors['B'], self.output_tensors['Y']
        A_local = get_local_tensor_info(A_info, intra_sbp_sigs['A'])
        B_local = get_local_tensor_info(B_info, intra_sbp_sigs['B'])
        Y_local = get_local_tensor_info(Y_info, intra_sbp_sigs['Y'])
        shape_A_transpose = transpose(A_local.shape, -2, -1)
        shape_B_transpose = transpose(B_local.shape, -2, -1)
        bp_A_sram_access = self.__get_matmul_sram_access_count(shape_A_transpose, Y_local.shape)
        bp_B_sram_access = self.__get_matmul_mac_count(Y_local.shape, shape_B_transpose)
        return self._estimate_latency_with_roofline_model(self.__get_bp_A_mac_count(), bp_A_sram_access, arch_config) \
             + self._estimate_latency_with_roofline_model(self.__get_bp_B_mac_count(), bp_B_sram_access, arch_config) 

    def _generate_candidate_intra_sbp_sigs(self) -> None:
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
                          PartialSbpParallel(),
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
        
        self._candidate_intra_sbp_sigs = candidate_sbp_signatures
    
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
    
    @property
    def _dim_table(self) -> pd.DataFrame:
        A_info, B_info, Y_info = self.input_tensors['A'], self.input_tensors['B'], self.output_tensors['Y']
        A_shape_dims, B_shape_dims, Y_shape_dims = len(A_info.shape), len(B_info.shape), len(Y_info.shape)

        index = [f"B{i}" for i in range(Y_shape_dims - 2)] + ['M', 'K', 'N']
        data = {
            'A': [np.nan] * (Y_shape_dims - A_shape_dims) + list(range(A_shape_dims - 2)) + [B_shape_dims - 2, A_shape_dims - 1, np.nan],
            'B': [np.nan] * (Y_shape_dims - B_shape_dims) + list(range(B_shape_dims - 2)) + [np.nan, B_shape_dims - 2, B_shape_dims - 1],
            'Y': list(range(Y_shape_dims - 2)) + [Y_shape_dims - 2, np.nan, Y_shape_dims - 1],
        }

        return pd.DataFrame(data, index=index)
    
def get_linear_testcase():
    input_tensors = {
        'A': TensorInfo(
            name='test_A',
            shape=(1, 512, 768),
            onnx_dtype=10,
            kind='input',
            inplace=True,
        ),
        'B': TensorInfo(
            name='test_B',
            shape=(768, 768),
            onnx_dtype=10,
            kind='weight',
            inplace=True
        ),
    }
    output_tensors = {
        'Y': TensorInfo(
            name='test_Y',
            shape=(1, 512, 768),
            onnx_dtype=10,
            kind='activation',
            inplace=True
        )
    }
    return input_tensors, output_tensors

def get_attention_testcase():
    input_tensors = {
        'A': TensorInfo(
            name='test_A',
            shape=(1, 12, 512, 64),
            onnx_dtype=10,
            kind='input',
            inplace=False,
        ),
        'B': TensorInfo(
            name='test_B',
            shape=(1, 12, 64, 512),
            onnx_dtype=10,
            kind='input',
            inplace=False,
        ),
    }
    output_tensors = {
        'Y': TensorInfo(
            name='test_Y',
            shape=(1, 12, 512, 512),
            onnx_dtype=10,
            kind='activation',
            inplace=False,
        )
    }
    return input_tensors, output_tensors
    
if __name__ == "__main__":
    input_tensors, output_tensors = get_attention_testcase()
    matmul_op = MatMulOperator(
        'test_matmul',
        'MatMul',
        input_tensors=input_tensors,
        output_tensors=output_tensors
    )
    matmul_op.num_core_range = list(range(16384, 16384 + 1))
    matmul_op.generate_candidate_intra_sbp_sigs()

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
    training_config = TrainingConfig()
    matmul_op.debug = True
    best_sbp = matmul_op.find_best_sbp_signature(arch_config=arch_config, training_config=training_config)