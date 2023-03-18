import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import sqrt
from op_graph.op import MatMulOperator
from utils import multidirectional_broadcasting, TensorInfo, ArchConfig

# 8 x 8 GPU-like arch configuration
# Uniform all performance to 1GHz, that is, a 1 GHz WSE core achieves the same performance as a GPU
# A100 architecture info: 
# https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
arch_config = ArchConfig({
    # A100 freq = 2.821 GHz
    # A100 has 6912 FP32 CUDA core -> 19.5 TFLOPS,
    #          432  FP16 8x4x8 Tensor Core -> 312 TFLOPS
    'core_num_mac': 312000,      # A100 FP16 312 TFLOPS
    'core_buffer_width': 1555,   # A100 peak memory bandwidth 1555 GB/S
    'core_buffer_size': 80e9,    # A100 80G memory
    'noc_bandwidth': 600,        # NVLink 600 GB/s
    'core_array_height': 8,
    'core_array_width': 8,
    'reticle_array_height': 1,
    'reticle_array_width': 1,
})

# A100 has 164 KB shared memory per SM, and can be utilized to increase operational intensity
DRAM_BLOCKING_SIZE = int(sqrt(164000 / 3 / 2))  # 165

# Each SM equivalently has 2888 mac,
# and a 164KB shared memory with 32 banks * 16 bit / cycle bandwidth on every bank
# on an equivalent WSE core, it has 180.5 Byte / cycle bandwidth
# its maximum intensity = 16

# Each A100 SM has 65536 FP32 register, so we can set SHARED_MEMORY_BLOCKING_SIZE = 64
# Operational intensity = 165^3 / (2 * 2 * 165^3 / 64 + 165^2) = 16
# and is compute-bound

# Tensor Core is obviously compute-bound

# Therefore, we only need to analyze if we are stuck on DRAM bandwidth

def test_matmul(A_shape, B_shape):
    M_dim_value, N_dim_value = A_shape[-2], B_shape[-1]
    Y_shape = multidirectional_broadcasting(A_shape[:-2], B_shape[:-2]) + [M_dim_value, N_dim_value]

    # FP16 operation
    input_tensors = {
        'A': TensorInfo(A_shape, 10,'test_A'),
        'B': TensorInfo(B_shape, 10,'test_B'),
    }
    output_tensors = {
        'Y': TensorInfo(Y_shape, 10,'test_Y'),
    }
    op = MatMulOperator(
        'test_matmul',
        'MatMul',
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        M_block_size=DRAM_BLOCKING_SIZE,
        K_block_size=DRAM_BLOCKING_SIZE,
        N_block_size=DRAM_BLOCKING_SIZE,
    )
    op.num_core_range = [64]
    op.generate_candidate_sbp_signatures()
    op.find_best_sbp_signature(arch_config)

if __name__ == "__main__":
    test_matmul([100e3, 100e3], [100e3, 100e3])