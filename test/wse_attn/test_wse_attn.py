import pandas as pd
import argparse
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import traceback

# from dse4wse.model.wse_attn import WseTransformerRunner
from dse4wse.model.wse_attn import ReticleFidelityWseTransformerRunner as WseTransformerRunner
from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.utils import logger, TrainingConfig

logger.info(f"Transformer type: {WseTransformerRunner}")

parser = argparse.ArgumentParser()
parser.add_argument('--case', metavar='N', type=int, default=6)

def create_a100_like_wse(**kwargs):
    T = kwargs.get('tensor_parallel_size')
    P = 8  # pipeline stage on one wafer

    # instantiate wafer scale engine
    core_config = {
        'core_compute_power': 312e12,
        'core_sram_size': 48e6,
    }
    reticle_config = {  # 1 GPU
        'core_array_height': 1,
        'core_array_width': 1,
        'inter_core_bandwidth': 1,  # doesn't matter here
        'core_config': core_config,
    }
    wse_config = {   # 1 node
        'reticle_array_height': T,
        'reticle_array_width': P,
        'inter_reticle_bandwidth': 600e9,
        'dram_size': T * P * 80e9,
        'dram_bandwidth': 1.94e12,
        'dram_stacking_type': '2d',
        'reticle_config': reticle_config,
    }
    wafer_scale_engine = WaferScaleEngine(**wse_config)
    return wafer_scale_engine

def find_best_micro_batch_size(**kwargs) -> int:
    for handler in logger.handlers:
        handler.setLevel('WARNING')

    best_training_throughput = 0
    best_micro_batch_size = -1
    candidate_micro_batch_size = [2 ** i for i in range(int(math.log2(kwargs.get('mini_batch_size'))) + 1)]

    for micro_batch_size in candidate_micro_batch_size:
        wse_transformer_runner = WseTransformerRunner(
            attention_heads=kwargs.get('attention_heads'),
            hidden_size=kwargs.get('hidden_size'),
            sequence_length=kwargs.get('sequence_length'),
            number_of_layers=kwargs.get('number_of_layers'),
            micro_batch_size=micro_batch_size,
            mini_batch_size=kwargs.get('mini_batch_size'),
            data_parallel_size=kwargs.get('data_parallel_size'),
            model_parallel_size=kwargs.get('model_parallel_size'),
            tensor_parallel_size=kwargs.get('tensor_parallel_size'),
            wafer_scale_engine=kwargs.get('wafer_scale_engine'),
            inter_wafer_bandwidth=kwargs.get('inter_wafer_bandwidth'),
            training_config=kwargs.get('training_config'),
        )
        try:
            assert wse_transformer_runner.get_dram_utilization()
            throughput = wse_transformer_runner.get_training_throughput()
            if throughput > best_training_throughput:
                best_training_throughput = throughput
                best_micro_batch_size = micro_batch_size
        except:
            logger.warning(traceback.format_exc())
            continue
    if best_micro_batch_size == -1:
        logger.error("Failed to find valid micro batch size")
        exit(1)
    for handler in logger.handlers:
        handler.setLevel('DEBUG')

    return best_micro_batch_size

def test_attention_module(**kwargs):
    wafer_scale_engine = create_a100_like_wse(
        tensor_parallel_size=kwargs.get('tensor_parallel_size'),
    )
    training_config = TrainingConfig()
    inter_wafer_bandwidth = 25e9 * kwargs.get('tensor_parallel_size')  # infiniband
    best_micro_batch_size = find_best_micro_batch_size(
        wafer_scale_engine=wafer_scale_engine,
        training_config=training_config,
        inter_wafer_bandwidth=inter_wafer_bandwidth,
        **kwargs,
    )
    logger.info(f"best micro batch size: {best_micro_batch_size}")

    wse_transformer_runner = WseTransformerRunner(
        attention_heads=kwargs.get('attention_heads'),
        hidden_size=kwargs.get('hidden_size'),
        sequence_length=kwargs.get('sequence_length'),
        number_of_layers=kwargs.get('number_of_layers'),
        micro_batch_size=best_micro_batch_size,
        mini_batch_size=kwargs.get('mini_batch_size'),
        data_parallel_size=kwargs.get('data_parallel_size'),
        model_parallel_size=kwargs.get('model_parallel_size'),
        tensor_parallel_size=kwargs.get('tensor_parallel_size'),
        wafer_scale_engine=wafer_scale_engine,
        inter_wafer_bandwidth=inter_wafer_bandwidth,
        training_config=training_config,
    )

    throughput = wse_transformer_runner.get_training_throughput()
    wse_utilization = wse_transformer_runner.get_training_wse_utilization()
    wse_transformer_runner.get_dram_utilization()

    logger.info(f"Throughput: {throughput:.2f} sequence / second")
    logger.info(f"WSE utilization: {wse_utilization:.2%}")

def run_testcase(case):
    df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'megatron.xlsx'))
    megatron_config = df.iloc[case]

    number_of_parameters = megatron_config['number_of_parameters']
    logger.info(f"Running megatron test case {case} with {number_of_parameters} billion parameters")

    megatron_config = megatron_config.to_dict()
    for index in ['attention_heads', 'hidden_size', 'number_of_layers', 'tensor_parallel_size', 'model_parallel_size', 'number_of_gpus', 'mini_batch_size']:
        megatron_config[index] = int(megatron_config[index])
    megatron_config['sequence_length'] = 2048
    megatron_config['data_parallel_size'] = megatron_config['number_of_gpus'] // megatron_config['model_parallel_size'] // megatron_config['tensor_parallel_size']

    test_attention_module(**megatron_config)

if __name__ == "__main__":
    run_testcase(9)
    for i in range(9, 10):
        try:
            run_testcase(i)
        except:
            logger.warning(f"Failure in experiment {i}")
            logger.error(traceback.format_exc())
            exit(1)