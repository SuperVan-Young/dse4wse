# DSE4WSE
Design Space Exploration toolset for Wafer-Scale Engine

# Getting started

## Installing
Create a new python venv or conda environment and run:
```
python -m pip install -r requirements
```

## Operator Graph Estimation
```
python test/attention/test_attention.py --case 0
```

## PE Graph Estimation

We provide evaluator for DSE frameworks in `test/dse/api.py`, in which you can find detailed description.

## Use trained GNN model
Put your GNN model in test/test_gnn/checkpoint/