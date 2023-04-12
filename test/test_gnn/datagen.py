import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Union, Dict
import pickle as pkl
import numpy as np
import random

from dse4wse.model.wse_attn import ReticleFidelityWseTransformerRunner
from dse4wse.utils import TrainingConfig, logger
from dse4wse.pe_graph.task import (
    ListWaferTask, 
)
from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.pe_graph.mapper import get_default_mapper
from dse4wse.pe_graph.evaluator import LpReticleLevelWseEvaluator
from dse4wse.gnn.dataloader import LinkUtilDataset

class GnnDataGenTransformerRunner(ReticleFidelityWseTransformerRunner):
    def __init__(self, attention_heads: int, hidden_size: int, sequence_length: int, number_of_layers: int, micro_batch_size: int, mini_batch_size: int, data_parallel_size: int, model_parallel_size: int, tensor_parallel_size: int, wafer_scale_engine: WaferScaleEngine, training_config: TrainingConfig, inter_wafer_bandwidth: Union[int, None] = None, zero_dp_os: bool = True, zero_dp_g: bool = True, zero_dp_p: bool = False, zero_r_pa: bool = True, num_reticle_per_pipeline_stage: int = 1, **kwargs) -> None:
        super().__init__(attention_heads, hidden_size, sequence_length, number_of_layers, micro_batch_size, mini_batch_size, data_parallel_size, model_parallel_size, tensor_parallel_size, wafer_scale_engine, training_config, inter_wafer_bandwidth, zero_dp_os, zero_dp_g, zero_dp_p, zero_r_pa, num_reticle_per_pipeline_stage, **kwargs)

    def get_gnn_training_data(self, forward=True):
        """ Modified from get_propagation_latency
        """
        assert self.is_overlap

        # em, I have to use mangled names here
        input_task_list = self._ReticleFidelityWseTransformerRunner__assign_input_reticle_task(forward)
        need_to_swap_weight, need_to_swap_activation = self.get_sram_utilization(forward)
        swap_weight_task_list = self._ReticleFidelityWseTransformerRunner__assign_swap_weight_reticle_task(forward) if need_to_swap_weight else []
        swap_activation_task_list = self._ReticleFidelityWseTransformerRunner__assign_swap_activation_reticle_task(forward) if need_to_swap_activation else []
        compute_task_list = self._ReticleFidelityWseTransformerRunner__assign_compute_reticle_task(forward)
        allreduce_task_list = self._ReticleFidelityWseTransformerRunner__assign_allreduce_reticle_task(forward) if self.tensor_parallel_size != 1 else []

        task_lists = {
            'input': input_task_list,
            'swap_weight': swap_weight_task_list,
            'swap_activation': swap_activation_task_list,
            'compute': compute_task_list,
            'allreduce': allreduce_task_list,
        }

        task_list = sum(task_lists.values(), [])
        wse_task = ListWaferTask(task_list)
        mapper = get_default_mapper(self.wafer_scale_engine, wse_task)
        wse_evaluator = LpReticleLevelWseEvaluator(self.wafer_scale_engine, wse_task, mapper)
        
        gnn_training_data = wse_evaluator.dump_graph()
        return gnn_training_data
    
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
) -> WaferScaleEngine:
    """ Variable naming follows previous conventions
    """
    WSE_FREQUENCY = 1e9
    core_config = {
        'core_compute_power': core_mac_num * WSE_FREQUENCY,
        'core_sram_size': core_buffer_size * 1e3, # KB
    }
    reticle_config = {
        'core_array_height': core_array_h,
        'core_array_width': core_array_w,
        'inter_core_bandwidth': core_noc_bw * WSE_FREQUENCY,
        'core_config': core_config,
    }
    wse_config = {
        'reticle_array_height': reticle_array_h,
        'reticle_array_width': reticle_array_w,
        'inter_reticle_bandwidth': reticle_bw * WSE_FREQUENCY,
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
    num_reticle_per_pipeline_stage: int = 1,
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
        num_reticle_per_pipeline_stage=num_reticle_per_pipeline_stage,
    )

    return wse_transformer_runner

def generate_single_gnn_training_data(design_point: Dict, model_parameters: Dict):
    logger.info(f"Design point: {design_point}")
    logger.info(f"Model parameters: {model_parameters}")

    wafer_scale_engine = create_wafer_scale_engine(**design_point)
    evaluator = create_evaluator(wafer_scale_engine, **model_parameters)
    training_data = evaluator.get_gnn_training_data()

    return training_data

def generate_batch_gnn_training_data(idx_range=None):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_points.pickle"), 'rb') as f:
        legal_points = pkl.load(f)
    random.shuffle(legal_points, lambda : 0.42)  # fixed random seed
    if idx_range == None:
        idx_range = range(len(legal_points))
    for i in idx_range:
        design_point, model_parameters = legal_points[i]
        try:
            training_data = generate_single_gnn_training_data(design_point, model_parameters)
            # print(training_data)
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"{i}.pickle"), 'wb') as f:
                pkl.dump(training_data, f)
        except KeyboardInterrupt:
            exit(1)
        except:
            continue

def test_dataloader():
    dataset = LinkUtilDataset(save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    test_data = dataset[0]
    logger.debug(test_data)

if __name__ == "__main__":
    generate_batch_gnn_training_data(idx_range=None)
    test_dataloader()