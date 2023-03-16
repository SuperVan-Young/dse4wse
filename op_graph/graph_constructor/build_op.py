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
                   input_tensors: List[TensorInfo], 
                   output_tensors: List[TensorInfo]) -> Operator:
    
    if op_type == 'Log':
        input_tensors = {'in': input_tensors[0]}
        output_tensors = {'out': output_tensors[0]}
        op = UnaryElementwiseOperator(name, op_type, input_tensors, output_tensors, operation_intensity=1)
    elif op_type == 'Sqrt':
        input_tensors = {'in': input_tensors[0]}
        output_tensors = {'out': output_tensors[0]}
        op = UnaryElementwiseOperator(name, op_type, input_tensors, output_tensors, operation_intensity=10)
    else:
        raise NotImplementedError
    return op
