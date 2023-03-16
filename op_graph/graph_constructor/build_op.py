import os
import sys
from typing import List, Dict, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from op_graph.op import (
    Operator,
    UnaryElementwiseOperator
)
from utils import logger, TensorInfo

def build_operator(name: str, op_type: str, 
                   input_tensors: Dict[str, TensorInfo], 
                   output_tensors: Dict[str, TensorInfo]) -> Operator:
    
    if op_type == 'log':
        op = UnaryElementwiseOperator(name, op_type, input_tensors, output_tensors, operation_intensity=1)
    elif op_type == 'sqrt':
        op = UnaryElementwiseOperator(name, op_type, input_tensors, output_tensors, operation_intensity=10)
    else:
        raise NotImplementedError
    return op
