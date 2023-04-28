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
import time

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

from api import create_evaluator, create_gnn_evaluator, create_wafer_scale_engine

class Timer:
    def __enter__(self):
        logger.info(f"Entering timer context")
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        logger.info(f"Elapsed time: {self.elapsed_time:.6f} seconds")

def build_legal_points(benchmark_size):
    # build a list of design point and model parameters
    legal_points = []
    design_point_source = 'random'
    logger.info(f"design point source: {design_point_source}")

    if design_point_source in ['gnn_train_data', 'gnn_test_data']:
        suffix = 'train' if design_point_source == 'gnn_train_data' else 'test'
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_gnn', 'data', suffix)
        dataset = NoCeptionDataset(save_dir=save_dir)
        legal_points = [(data['design_point'], data['model_parameters']) for data in dataset]
    elif design_point_source == 'random':
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_gnn", "legal_points.pickle"), 'rb') as f:
            legal_points = pkl.load(f)
        random.shuffle(legal_points, lambda : 0.73)  # different seed from building dataset
    else:
        raise NotImplementedError

    # we manually divide design points by benchmark workload size
    # benchmark_size_2_hidden_size = {
    #     'small': [2304, 3072, 4096, 6144, 8192],
    #     'medium': [10240, 12288, 16384, 20480, 25600],
    #     'large': [32000, 43200, 66560, 80600, 102000, 158720],
    # }
    benchmark_size_2_hidden_size = {
        0: [2304],
        1: [3072],
        2: [4096],
        3: [6144],
        4: [8192],
        5: [10240],
        6: [12288],
        7: [16384],
        8: [20480],
        9: [25600],
        10: [32000],
        11: [43200],
        12: [66560],
        13: [80600],
        14: [102000],
    }
    hidden_sizes = benchmark_size_2_hidden_size[benchmark_size]
    legal_points = [lp for lp in legal_points if lp[1]['hidden_size'] in hidden_sizes]
    logger.debug(f"Number of legal points: {len(legal_points)}")

    return legal_points

def test_fidelity_accuracy(fidelity, legal_points):
    """ test the accuracy of naive & GNN fidelity against LP solver
    """
    logger.info(f"Testing fidelity {fidelity}'s accuracy against LP solver with {len(legal_points)} legal points")

    # preload pretrained gnn model for better efficiency. For now we hardcode the best one we get

    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_gnn', 'checkpoint', "model_2023-04-29-00-29-43-272787.pth")
    gnn_model = torch.load(checkpoint_path)
    gnn_model.eval()

    # now let's test some transformer runner!
    baseline_results = []
    test_results = []

    def worker(design_point, model_parameters, fidelity='baseline') -> int:
        for handler in logger.handlers:
            handler.setLevel('WARNING')

        wafer_scale_engine = create_wafer_scale_engine(**design_point)
        evaluator = None
        if fidelity == 'baseline':
            evaluator = create_evaluator(True, wafer_scale_engine, **model_parameters)
        elif fidelity == 'naive':
            evaluator = create_evaluator(False, wafer_scale_engine, **model_parameters)
        elif fidelity == 'gnn':
            evaluator = create_gnn_evaluator(gnn_model, wafer_scale_engine, **model_parameters)
        else:
            raise NotImplementedError
        try:
            result = evaluator.get_training_throughput()
        except:
            logger.warning("Failure in getting training throughput")
            # 100% fidelity
            result = 0

        for handler in logger.handlers:
            handler.setLevel('DEBUG')
        return result

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
    timer = Timer()
    with timer:
        for design_point, model_parameters in tqdm_bar:
            result = worker(design_point, model_parameters, fidelity=fidelity)
            test_results.append(result)
    
    tqdm_bar = tqdm(legal_points)
    for design_point, model_parameters in tqdm_bar:
        result = worker(design_point, model_parameters, fidelity='baseline')
        baseline_results.append(result)

    # analyze the result
    logger.info(f"MAE = {get_mae()}")
    logger.info(f"MAPE = {get_mape()}")

    report = {
        'average_time': timer.elapsed_time / len(legal_points),
        'accuracy': get_mape(),
    }
    return report

def get_simulation_elapsed_time(legal_points):
    total_elapsed_time = []

    for handler in logger.handlers:
        handler.setLevel('WARNING')

    tqdm_bar = tqdm(legal_points)
    for design_point, model_parameters in tqdm_bar:
        wafer_scale_engine = create_wafer_scale_engine(**design_point)
        evaluator = create_evaluator(False, wafer_scale_engine, **model_parameters)  # use naive version first
        elapsed_time = evaluator.get_simulation_elapsed_time()
        total_elapsed_time.append(elapsed_time)

    for handler in logger.handlers:
        handler.setLevel('DEBUG')

    average_elapsed_time = np.mean(total_elapsed_time).item()
    logger.info(average_elapsed_time)

    return average_elapsed_time

def main():
    EVALUATION_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation')
    if not os.path.exists(EVALUATION_REPORT_DIR):
        os.mkdir(EVALUATION_REPORT_DIR)

    # for fidelity in ['naive', 'gnn']:
    for fidelity in ['gnn']:
        df = pd.DataFrame(columns=['average_time', 'accuracy'])
        for benchmark_size in range(15):
            legal_points = build_legal_points(benchmark_size=benchmark_size)
            report = test_fidelity_accuracy(fidelity=fidelity, legal_points=legal_points)
            df.loc[len(df.index)] = report
        df.to_csv(os.path.join(EVALUATION_REPORT_DIR, f"{fidelity}.csv"), index=True)
    exit(1)

    # simulation results 
    df = pd.DataFrame(columns=['average_time', 'accuracy'])
    for benchmark_size in range(15):
        legal_points = build_legal_points(benchmark_size=benchmark_size)
        average_time = get_simulation_elapsed_time(legal_points)
        report = {
            'average_time': average_time,
            'accuracy': 0,
        }
        df.loc[len(df.index)] = report
    df.to_csv(os.path.join(EVALUATION_REPORT_DIR, f"simulation.csv"), index=True)


if __name__ == "__main__":
    main()