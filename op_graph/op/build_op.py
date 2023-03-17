import os
import sys
from typing import List, Dict, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import logger, TensorInfo

from base import Operator
from unary_elementwise import UnaryElementwiseOperator
from binary_elementwise import BinaryElementwiseOperator
from matmul import MatMulOperator

def build_operator(name: str, op_type: str, 
                   input_tensors: List[TensorInfo], 
                   output_tensors: List[TensorInfo]) -> Operator:
    
    # Unary Elementwise Operators
    if op_type == 'Log':
        input_tensors = {'in': input_tensors[0]}
        output_tensors = {'out': output_tensors[0]}
        op = UnaryElementwiseOperator(name, op_type, input_tensors, output_tensors, operation_intensity=1)
    elif op_type == 'Sqrt':
        input_tensors = {'in': input_tensors[0]}
        output_tensors = {'out': output_tensors[0]}
        op = UnaryElementwiseOperator(name, op_type, input_tensors, output_tensors, operation_intensity=10)
    
    # Binary Elementwise Operators
    elif op_type in ['Add', 'Sub', 'Mul', 'Div', 'Pow']:
        input_tensors = {
            'A': input_tensors[0],
            'B': input_tensors[1],
        }
        output_tensors = {'out': output_tensors[0]}
        #TODO: measure realy operation intensity
        op = BinaryElementwiseOperator(name, op_type, input_tensors, output_tensors, operation_intensity=1)

    # MatMul
    elif op_type in ['MatMul']:
        # ignore adding in Gemm
        input_tensors = {
            'A': input_tensors[0],
            'B': input_tensors[1],
        }
        output_tensors = {'Y': output_tensors[0]}
        op = MatMulOperator(name, op_type, input_tensors, output_tensors)

    else:
        raise NotImplementedError
    return op
