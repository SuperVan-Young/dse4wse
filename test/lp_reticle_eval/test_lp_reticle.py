
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dse4wse.op_graph.op import BaseOperator, MatMulOperator

from dse4wse.pe_graph.evaluator import LpReticleLevelWseEvaluator
from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.pe_graph.task import ListWaferTask, ThreeStageReticleTaskGenerator
from dse4wse.pe_graph.mapper import get_default_mapper

from dse4wse.utils import TensorInfo, logger

def instantiate_wafer(**kwargs):
    core_config = {
        'core_num_mac': kwargs.get('core_num_mac'),
    }
    reticle_config = {
        'core_array_height': 1,
        'core_array_width': 1,
        'inter_core_bandwidth': 1,  # doesn't matter here
        'core_config': core_config,
    }
    wse_config = {
        'reticle_array_height': kwargs.get('tensor_parallel_size'),
        'reticle_array_width': 1,
        'inter_reticle_bandwidth': kwargs.get('inter_reticle_bandwidth'),
        'dram_bandwidth': kwargs.get('dram_bandwidth'),
        'dram_stacking_type': kwargs.get('dram_stacking_type'),
        'reticle_config': reticle_config,
    }
    wse = WaferScaleEngine(**wse_config)
    return wse

def create_task_generator_from_operator(op: BaseOperator):
    compute_amount = op.get_fp_mac_count()
    read_data_amount = [tensor.size() for tensor in op.input_tensors.values()]
    write_data_amount = [tensor.size() for tensor in op.output_tensors.values()]
    kwargs = {
        'compute_amount': compute_amount,
        'read_data_amount': read_data_amount,
        'write_data_amount': write_data_amount,
        'new_dram_port': True,
    }
    return ThreeStageReticleTaskGenerator(**kwargs)

def instantiate_task(**kwargs):
    B, S, H = kwargs.get('micro_batch_size'), kwargs.get('sequence_length'), kwargs.get('hidden_size')
    H_ = H // kwargs.get('tensor_parallel_size')
    BFLOAT16 = 10

    X = TensorInfo(
        name='X',
        shape=(B, S, H),
        onnx_dtype=BFLOAT16,
        kind='input',
        inplace=False,
    )
    W_qkv = TensorInfo(
        name='W_qkv',
        shape=(H, 3 * H_),
        onnx_dtype=BFLOAT16,
        kind='weight',
        inplace=True,
    )
    QKV = TensorInfo(
        name='QKV',
        shape=(B, S, 3 * H_),
        onnx_dtype=BFLOAT16,
        kind='activation',
        inplace=False,
    )
    linear_qkv = MatMulOperator(
        name='linear_qkv',
        op_type='Matmul',
        input_tensors={'A': X, 'B': W_qkv},
        output_tensors={'Y': QKV},
    )
    task_generator = create_task_generator_from_operator(linear_qkv)

    wse_task = ListWaferTask([task_generator(repeated_times=6) for _ in range(kwargs.get('tensor_parallel_size'))])

    return wse_task

def test_lp_reticle_evaluator(**kwargs):
    hardware = instantiate_wafer(**kwargs)
    task = instantiate_task(**kwargs)
    mapper = get_default_mapper(hardware, task)

    wse_evaluator = LpReticleLevelWseEvaluator(
        hardware=hardware,
        task=task,
        mapper=mapper,
    )
    total_latency = wse_evaluator.get_total_latency()
    logger.info(f"Total latency {total_latency} seconds")

    return total_latency

TESTCASE = {
    'core_num_mac': 312e12,
    'inter_reticle_bandwidth': 600e9,
    'dram_bandwidth': 1.94e12,
    'dram_stacking_type': '2d',

    'micro_batch_size': 32,
    'sequence_length': 2048,
    'hidden_size': 12288,
    'tensor_parallel_size': 8,
}

if __name__ == "__main__":
    test_lp_reticle_evaluator(**TESTCASE)