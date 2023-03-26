import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import argparse

from op_graph.module import AttentionModule
from utils import ArchConfig, logger

parser = argparse.ArgumentParser()
parser.add_argument('--case', metavar='N', type=int, default=6)

# Megatron uses #Tensor_parallel x #data_parallel GPU for one pipeline stage (roughly 1~2 layers)
# We use similar setup for the model size

def test_attention_module(
    attention_heads: int,
    hidden_size: int,
    sequence_length: int,
    micro_batch_size: int,
    mini_batch_size: int,
    number_of_layers: int,  # total
    data_parallel_size: int,
    model_parallel_size: int,
    tensor_parallel_size: int,
    **kwargs,
    ):
    swap_weight_timesteps = attention_heads

    attention_module = AttentionModule(
        attention_heads = attention_heads,
        hidden_size = hidden_size,
        sequence_length = sequence_length,
        micro_batch_size = micro_batch_size,
        mini_batch_size = mini_batch_size,
        number_of_layers = number_of_layers,
        data_parallel_size = data_parallel_size,
        model_parallel_size = model_parallel_size,
        tensor_parallel_size = tensor_parallel_size,
        swap_weight_timesteps = swap_weight_timesteps,
    )
    
    # similar hardware resources like Megatron-2 paper setup for corresponding model
    # assume 1Ghz frequency
    # FIXME: one more thing, we assume matmul blocking size = 64 in default
    # This is not always correct for all WSE chips
    # We should take number of registers into design consideration
    arch_config = ArchConfig({
        'core_frequency': 1e9,               # 1GHz
        'core_num_mac': 312000,              # A100 FP16 312 TFLOPS
        'core_num_reg': 32 * 1024 * 108,     # 64K / SM and 108 SM, we assume we only use 50%
        'core_sram_size': 40 * 1024 * 1024,  # 40MB L2 cache
        'core_sram_bandwidth': 4830,         # 2.3x 2.1TB/s (V100)
        'inter_core_bandwidth': 7 * 600,     # same with reticle    
        'core_array_height': 1,
        'core_array_width': 1,
        'inter_reticle_bandwidth': 7 * 600,  # 7x NVLink 3.0 x 12 ports
        'reticle_array_height': 1,           # data parallel size is considered within attn module
        'reticle_array_width': int(tensor_parallel_size),
        'wafer_dram_size': data_parallel_size * tensor_parallel_size * 80e9,
        'wafer_dram_bandwidth': 1.94e3,      # 1.94TB/s HBM      
        'wafer_dram_stacking_type': '2d',
        'inter_wafer_bandwidth': 25,         # 200Gb/s infiniband
    })

    attention_module.alloc_core_and_derive_sbp_sig(arch_config)

    logger.warning("Using GPU-like mode for attention module")
    attention_module.gpu_like = True

    training_throughput = attention_module.get_training_throughput(arch_config)

    logger.info(f"Training throughput: {training_throughput} sequence / second")

def run_testcase(case=1):
    df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'megatron.xlsx'))
    megatron_config = df.iloc[case]

    number_of_parameters = megatron_config['number_of_parameters']
    logger.info(f"Running megatron test case {case} with {number_of_parameters} billion parameters")

    megatron_config = megatron_config.to_dict()
    for index in ['attention_heads', 'hidden_size', 'number_of_layers', 'tensor_parallel_size', 'model_parallel_size', 'number_of_gpus', 'mini_batch_size']:
        megatron_config[index] = int(megatron_config[index])
    megatron_config['sequence_length'] = 2048
    megatron_config['micro_batch_size'] = 1 # the smallest size ...
    megatron_config['data_parallel_size'] = megatron_config['number_of_gpus'] // megatron_config['model_parallel_size'] // megatron_config['tensor_parallel_size']

    test_attention_module(**megatron_config)

if __name__ == "__main__":
    # args = parser.parse_args()
    # run_testcase(args.case)
    for i in range(10):
        try:
            run_testcase(i)
        except:
            logger.warning(f"Failure in experiment {i}")