import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base import Operator
from utils import (
    ArchConfig, SbpSignature, TensorInfo, generate_split_from_fixed_shape
)

class BinaryElementwiseOperator(Operator):
    def __init__(self, name: str, op_type: str, input_tensors: Dict[str, TensorInfo], output_tensors: Dict[str, TensorInfo],
                 *args, **kwargs) -> None:
        super().__init__(name, op_type, input_tensors, output_tensors, *args, **kwargs)