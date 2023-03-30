from typing import List, Dict, Union

from dse4wse.utils import logger, TensorInfo

from .base import BaseOperator
from .unary_elementwise import UnaryElementwiseOperator
from .binary_elementwise import BinaryElementwiseOperator
from .matmul import MatMulOperator

def build_operator(name: str, op_type: str, 
                   input_tensors: List[TensorInfo], 
                   output_tensors: List[TensorInfo]) -> BaseOperator:
    
    # Unary Elementwise Operators
    if op_type in ['Log', 'Erf', 'Tanh']:
        input_tensors = {'in': input_tensors[0]}
        output_tensors = {'out': output_tensors[0]}
        op = UnaryElementwiseOperator(name, op_type, input_tensors, output_tensors, mac_per_element=1)
    elif op_type in ['Sqrt']:
        input_tensors = {'in': input_tensors[0]}
        output_tensors = {'out': output_tensors[0]}
        op = UnaryElementwiseOperator(name, op_type, input_tensors, output_tensors, mac_per_element=10)
    
    # Binary Elementwise Operators
    elif op_type in ['Add', 'Sub', 'Mul', 'Div', 'Pow']:
        input_tensors = {
            'A': input_tensors[0],
            'B': input_tensors[1],
        }
        output_tensors = {'out': output_tensors[0]}
        #TODO: measure realy operation intensity
        op = BinaryElementwiseOperator(name, op_type, input_tensors, output_tensors, mac_per_element=1)

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
