import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Union, Dict, List
import pickle as pkl
import numpy as np
import random
import torch
import multiprocessing

from dse4wse.model.wse_attn import ReticleFidelityWseTransformerRunner
from dse4wse.utils import TrainingConfig, logger
from dse4wse.pe_graph.task import (
    ListWaferTask, 
)
from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.pe_graph.mapper import get_default_mapper
from dse4wse.pe_graph.evaluator import LpReticleLevelWseEvaluator
from dse4wse.gnn.dataloader import NoCeptionDataset

class GnnDataGenTransformerRunner(ReticleFidelityWseTransformerRunner):
    def __init__(self, attention_heads: int, hidden_size: int, sequence_length: int, number_of_layers: int, micro_batch_size: int, mini_batch_size: int, data_parallel_size: int, model_parallel_size: int, tensor_parallel_size: int, wafer_scale_engine: WaferScaleEngine, training_config: TrainingConfig, inter_wafer_bandwidth: Union[int, None] = None, zero_dp_os: bool = True, zero_dp_g: bool = True, zero_dp_p: bool = False, zero_r_pa: bool = True, **kwargs) -> None:
        super().__init__(attention_heads, hidden_size, sequence_length, number_of_layers, micro_batch_size, mini_batch_size, data_parallel_size, model_parallel_size, tensor_parallel_size, wafer_scale_engine, training_config, inter_wafer_bandwidth, zero_dp_os, zero_dp_g, zero_dp_p, zero_r_pa, **kwargs)

    def get_gnn_training_data(self, inference: bool=False, num_data: int=1) -> List:
        """ Modified from get_propagation_latency
        """
        assert self.is_overlap

        self._find_best_intra_model_chunk_exec_params(inference=inference)
        task_lists = self._get_task_lists(inference=inference)
        task_list = sum(task_lists.values(), [])
        wse_task = ListWaferTask(task_list)
        mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
        wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
        
        vrids = wse_task.get_all_virtual_reticle_ids()
        if num_data > len(vrids): num_data = len(vrids)
        random.shuffle(vrids)
        gnn_training_data_list = []
        for i in range(num_data):
            vrid = vrids[i]
            gnn_training_data_list.append(wse_evaluator.dump_graph_v2(vrid))
        return gnn_training_data_list
    
def create_wafer_scale_engine(
    core_buffer_size: int,
    core_buffer_bw: int,
    core_mac_num: int,
    core_noc_bw: int,
    core_noc_vc: int,
    core_noc_buffer_size: int,
    reticle_bw: int,
    core_array_h: int,
    core_array_w: int,
    wafer_mem_bw: int,
    reticle_array_h: int,
    reticle_array_w: int,
    dram_stacking_type: str = '2d',
    **kwargs,
) -> WaferScaleEngine:
    """ Variable naming follows previous conventions
    """
    WSE_FREQUENCY = 1e9
    core_config = {
        'core_compute_power': core_mac_num * WSE_FREQUENCY,
        'core_sram_size': core_buffer_size * 1e3, # KB
        'core_buffer_bandwidth': core_buffer_bw * WSE_FREQUENCY,
        'core_noc_vc': core_noc_vc,
        'core_noc_buffer_size': core_noc_buffer_size,
    }
    reticle_config = {
        'core_array_height': core_array_h,
        'core_array_width': core_array_w,
        'inter_core_bandwidth': core_noc_bw * WSE_FREQUENCY / 8,
        'core_config': core_config,
    }
    wse_config = {
        'reticle_array_height': reticle_array_h,
        'reticle_array_width': reticle_array_w,
        'inter_reticle_bandwidth': reticle_bw * core_noc_bw * WSE_FREQUENCY / 8,  # 如果是相对带宽，要做一个转换！
        'dram_size': np.inf,  # ideally
        'dram_bandwidth': wafer_mem_bw * WSE_FREQUENCY,
        'dram_stacking_type': dram_stacking_type,
        'reticle_config': reticle_config,
    }
    wafer_scale_engine = WaferScaleEngine(**wse_config)
    return wafer_scale_engine

def create_evaluator(
    wafer_scale_engine: WaferScaleEngine,
    attention_heads: int,
    hidden_size: int,
    sequence_length: int,
    number_of_layers: int,
    mini_batch_size: int,
    micro_batch_size: int = 1,
    data_parallel_size: int = 1,
    model_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    **kwargs,
):
    """ Modified from dse/api.py same-name function
    """
    default_training_config = TrainingConfig()
    default_inter_wafer_bandwidth = None

    wse_transformer_runner = GnnDataGenTransformerRunner(
        attention_heads=attention_heads,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        number_of_layers=number_of_layers,
        mini_batch_size=mini_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
        model_parallel_size=model_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        wafer_scale_engine=wafer_scale_engine,
        training_config=default_training_config,
        inter_wafer_bandwidth=default_inter_wafer_bandwidth,
    )

    return wse_transformer_runner

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(TRAIN_DATA_DIR):
    os.mkdir(TRAIN_DATA_DIR)
if not os.path.exists(TEST_DATA_DIR):
    os.mkdir(TEST_DATA_DIR)

def generate_single_gnn_data(idx: int, design_point: Dict, model_parameters: Dict, data_dir: str) -> List:
    logger.info(f"Design point idx {idx}")
    # logger.info(f"Design point: {design_point}")
    # logger.info(f"Model parameters: {model_parameters}")

    wafer_scale_engine = create_wafer_scale_engine(**design_point)
    evaluator = create_evaluator(wafer_scale_engine, **model_parameters)
    training_data_list = evaluator.get_gnn_training_data(inference=False, num_data=1)
    training_data_list += evaluator.get_gnn_training_data(inference=True, num_data=1)

    for j, training_data in enumerate(training_data_list):
        with open(os.path.join(data_dir, f"{idx}_{j}.pickle"), 'wb') as f:
            pkl.dump(training_data, f)

def wrapper_generate_single_gnn_train_data(x):
    idx, (design_point, model_parameters) = x
    try:
        generate_single_gnn_data(idx, design_point, model_parameters, data_dir=TRAIN_DATA_DIR)
    except KeyboardInterrupt:
        exit(1)
    except:
        logger.debug("Error in generating data")

def wrapper_generate_single_gnn_test_data(x):
    idx, (design_point, model_parameters) = x
    try:
        generate_single_gnn_data(idx, design_point, model_parameters, data_dir=TEST_DATA_DIR)
    except KeyboardInterrupt:
        exit(1)
    except:
        logger.debug("Error in generating data")

def generate_batch_gnn_data(idx_range=None, multiprocess=False, training=True, num_worker=32):
    if training:
        seed = 0.42
        data_dir = TRAIN_DATA_DIR
        gen_func = wrapper_generate_single_gnn_train_data
    else:
        seed = 0.84
        data_dir = TEST_DATA_DIR
        gen_func = wrapper_generate_single_gnn_test_data
    for file in os.listdir(data_dir):
        os.remove(os.path.join(data_dir, file))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_points.pickle"), 'rb') as f:
        legal_points = pkl.load(f)
    random.shuffle(legal_points, lambda : seed)  # fixed random seed
    if idx_range == None:
        idx_range = len(legal_points)
    if multiprocess:
        with multiprocessing.Pool(num_worker) as pool:
            pool.map(gen_func, enumerate(legal_points[:idx_range]))
    else:
        for x in enumerate(legal_points[:idx_range]):
            gen_func(x)

def test_dataloader():
    dataset = NoCeptionDataset(save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    for data in dataset:
        logger.debug(data)

if __name__ == "__main__":
    generate_batch_gnn_data(idx_range=2000, multiprocess=True, training=True)
    generate_batch_gnn_data(idx_range=200, multiprocess=True, training=False)
    # test_dataloader()