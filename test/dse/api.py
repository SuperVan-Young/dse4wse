# Connect my code to 

import pandas as pd
import os
import sys
import traceback
import numpy as np
from typing import Dict
import random
import torch
import pickle as pkl
from typing import Tuple
import multiprocessing as mp
from tqdm import tqdm

# make sure dse4wse filefolder is in your PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dse4wse.model.wse_attn import WseTransformerRunner
from dse4wse.model.wse_attn import ReticleFidelityWseTransformerRunner
from dse4wse.model.wse_attn import GnnReticleFidelityWseTransformerRunner
from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.utils import logger, TrainingConfig
from dse4wse.gnn.model import NoCeptionNet
from dse4wse.gnn.dataloader import NoCeptionDataset

def create_wafer_scale_engine(
    core_buffer_size: int,
    core_buffer_bw: int,
    core_mac_num: int,
    core_noc_bw: int,
    core_noc_vc: int,
    core_noc_buffer_size: int,
    reticle_bw: float,  # ratio ... # 注意检查一下设计点，看看给的是相对带宽还是绝对带宽！
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
    use_high_fidelity: bool,  # 选一下用哪个fidelity！
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
    """ kwargs with initial values can be cherry-picked for specific workloads.

    Update: now you can assign arbitrary number of reticles for one pipeline stage.
    However, we also add more restrictions about on-chip SRAM utilization.
    SO MAKE SURE YOU USE PROPER CONFIGURATIONS in case of unexpected assertion errors.
    """
    default_training_config = TrainingConfig()
    default_inter_wafer_bandwidth = None

    transformer_runner = ReticleFidelityWseTransformerRunner if use_high_fidelity else WseTransformerRunner

    wse_transformer_runner = transformer_runner(
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

def create_gnn_evaluator(
    gnn_model,
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
    default_training_config = TrainingConfig()
    default_inter_wafer_bandwidth = None

    wse_transformer_runner = GnnReticleFidelityWseTransformerRunner(
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
        gnn_model=gnn_model,
    )
    return wse_transformer_runner


def nohup_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            exit(1)
        except:
            logger.warning(traceback.format_exc())
            return np.inf
    return wrapper

# @nohup_decorator
def evaluate_design_point(design_point: Dict, model_parameters: Dict, metric='power', use_high_fidelity: bool=True):
    """ Evaluator API for DSE framework. 
    """
    logger.info(f"Design point: {design_point}")
    logger.info(f"Model parameters: {model_parameters}")

    wafer_scale_engine = create_wafer_scale_engine(**design_point)
    evaluator = create_evaluator(use_high_fidelity, wafer_scale_engine, **model_parameters)

    result = None
    if metric == 'throughput':
        result = evaluator.get_training_throughput()
    elif metric == 'training_utilization':
        result = evaluator.get_training_wse_utilization()  # with useful debugging info
    elif metric == 'latency':
        result = evaluator.get_inference_latency()
    elif metric == 'power':
        result = evaluator.get_training_peak_power()
    else:
        raise NotImplementedError
    logger.info(f"{metric}: {result}")

    return result

def design_space_exploration():
    """ Try a design point and see if there's any bug.
    """
    df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "design_points.xlsx"))
    random.seed(42)

    for i in range(10):
        test_index = random.randint(0, len(df.index))
        test_design_point = df.loc[test_index].to_dict()
        test_model_parameters = {
            "attention_heads": 12,
            "hidden_size": 768,
            "sequence_length": 512,
            "number_of_layers": 24,
            "mini_batch_size": 512,
            "micro_batch_size": 32,
            "tensor_parallel_size": 1,
            "model_parallel_size": 24,
            "num_reticle_per_model_chunk": 10,
        }
        evaluate_design_point(design_point = test_design_point, model_parameters = test_model_parameters)

def test_fidelity_accuracy(fidelity='naive'):
    """ test the accuracy of naive & GNN fidelity against LP solver
    """
    logger.info(f"Testing fidelity {fidelity}'s accuracy against LP solver")

    # build a list of design point and model parameters
    legal_points = []
    design_point_source = 'gnn_train_data'
    logger.info(f"design point source: {design_point_source}")

    if design_point_source in ['gnn_train_data', 'gnn_test_data']:
        suffix = 'train' if design_point_source == 'gnn_train_data' else 'test'
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_gnn', 'data', suffix)
        dataset = NoCeptionDataset(save_dir=save_dir)
        legal_points = [(data['design_point'], data['model_parameters']) for data in dataset]
    elif design_point_source == 'random':
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_points.pickle"), 'rb') as f:
            legal_points = pkl.load(f)
        random.shuffle(legal_points, lambda : 0.73)  # different seed from building dataset
        legal_points = legal_points[:500]  # restrict number of design points
    else:
        raise NotImplementedError

    # preload pretrained gnn model for better efficiency. For now we hardcode the best one we get
    best_model_param = {
        'h_dim': 128,
        'n_layer': 3,
        'use_deeper_mlp_for_inp': True,
        'use_deeper_mlp_for_edge_func': True,
        'pooling': 'set2set',
    }
    gnn_model = NoCeptionNet(**best_model_param)

    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_gnn', 'checkpoint', "model_2023-04-24-05-40-30-946075.pth")
    checkpoint = torch.load(checkpoint_path)
    gnn_model.load_state_dict(checkpoint['model_state_dict'])

    # now let's test some transformer runner!
    baseline_results = []
    test_results = []

    def worker(design_point, model_parameters) -> Tuple[int, int]:
        wafer_scale_engine = create_wafer_scale_engine(**design_point)
        baseline_evaluator = create_evaluator(True, wafer_scale_engine, **model_parameters)

        test_evaluator = None
        if fidelity == 'naive':
            test_evaluator = create_evaluator(False, wafer_scale_engine, **model_parameters)
        elif fidelity == 'gnn':
            test_evaluator = create_gnn_evaluator(gnn_model, wafer_scale_engine, **model_parameters)
        else:
            raise NotImplementedError

        # TODO: more metrics
        baseline_result = baseline_evaluator.get_training_throughput()
        test_result = test_evaluator.get_training_throughput()

        return (baseline_result, test_result)

    def get_mae():
        baseline_results_ = np.array(baseline_results)
        test_results_ = np.array(test_results)
        mae = np.abs(baseline_results_ - test_results_)
        mae = np.mean(mae)
        return mae
    
    def get_mape():
        baseline_results_ = np.array(baseline_results)
        test_results_ = np.array(test_results)
        mape = np.abs((baseline_results_ - test_results_) / baseline_results_)
        mape = np.mean(mape)
        return mape
    
    tqdm_bar = tqdm(legal_points)
    for design_point, model_parameters in tqdm_bar:
        result = worker(design_point, model_parameters)
        baseline_results.append(result[0])
        test_results.append(result[1])
        tqdm_bar.set_description(f"MAPE = {get_mape():.4%}")

    # analyze the result
    logger.info(f"MAE = {get_mae()}")
    logger.info(f"MAPE = {get_mape()}")


if __name__ == "__main__":
    # design_space_exploration()
    test_fidelity_accuracy(fidelity='gnn')
    # test_fidelity_accuracy(fidelity='naive')