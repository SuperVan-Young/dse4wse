import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from run_model import run

parser = argparse.ArgumentParser()
parser.add_argument('--param', metavar='N', type=int)

MODEL_PARAMS = [
    # larger hidden size
    {
        'h_dim': 128,
    },
    # less layer
    {
        'n_layer': 1,
    },
    # more layer
    {
        'n_layer': 3,
    },
    {
        'use_deeper_mlp_for_inp': True,
    },
    {
        'use_deeper_mlp_for_edge_func': True,
    },
    {
        'pooling': 'set2set',
    },

    # combine more tricks
    {
        'use_deeper_mlp_for_inp': True,
        'use_deeper_mlp_for_edge_func': True,
    },
    {
        'use_deeper_mlp_for_inp': True,
        'use_deeper_mlp_for_edge_func': True,
        'pooling': 'set2set',
    },
    # combine all tricks
    {
        'h_dim': 128,
        'n_layer': 3,
        'use_deeper_mlp_for_inp': True,
        'use_deeper_mlp_for_edge_func': True,
        'pooling': 'set2set',
    },
]

if __name__ == "__main__":
    args = parser.parse_args()
    model_param = MODEL_PARAMS[args.param]
    run(model_param)