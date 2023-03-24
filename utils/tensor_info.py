
from typing import Tuple
import re
import numpy as np
from functools import reduce
from onnx.mapping import TENSOR_TYPE_MAP
from sbp import SbpSignature
from copy import deepcopy

def onnx_dtype_2_storage_size(dtype: int) -> int:
    name = TENSOR_TYPE_MAP[dtype].name
    size_pattern = re.compile(r"^TensorProto\.[A-Z]+(\d+)$")

    size_match = size_pattern.match(name)
    if size_match:
        size = int(size_match.group(1)) // 4
        return size
    else:
        if name == "TensorProto.FLOAT":
            return 4
        elif name == "TensorProto.DOUBLE":
            return 8
        elif name == "TensorProto.BOOL":
            return 1
        else:
            return np.NAN

class TensorInfo():
    def __init__(self, name: str, shape: Tuple, onnx_dtype: int, inplace=False) -> None:
        self.name = name
        self.shape = shape
        self.onnx_dtype = onnx_dtype
        self.inplace = inplace

    @property
    def dtype_size(self):
        return onnx_dtype_2_storage_size(self.onnx_dtype)
    
    def numel(self):
        return reduce(lambda x, y: x * y, self.shape, 1)  # shape could be empty for scalar value
    
    def size(self):
        """In Byte
        """
        return self.numel() * self.dtype_size
    
    def get_local_tensor_shape(self, sbp_signature: SbpSignature):
        shape_ = deepcopy(self.shape)
        shape_ = list(shape_)
        for dim_value, sbp_parallel in zip(sbp_signature.placement.shape, sbp_signature.sbp_parallels):
            if sbp_parallel.is_split():
                assert shape_[sbp_parallel.dim] % dim_value == 0
                shape_[sbp_parallel.dim] //= dim_value
        return shape_
    
def multidirectional_broadcasting(A_shape: Tuple[int], B_shape: Tuple[int]) -> Tuple[int]:
    """https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    """
    A_shape, B_shape = list(A_shape), list(B_shape)
    C_shape = []

    if len(A_shape) < len(B_shape):
        A_shape = [1] * (len(B_shape) - len(A_shape)) + A_shape
    elif len(A_shape) > len(B_shape):
        B_shape = [1] * (len(A_shape) - len(B_shape)) + B_shape

    for a, b in zip(A_shape, B_shape):
        if a == b:
            C_shape.append(a)
        else:
            assert a == 1 or b == 1
            C_shape.append(max(a, b))
    return C_shape

if __name__ == "__main__":
    print(onnx_dtype_2_storage_size(0))

    print(multidirectional_broadcasting([2, 3, 4, 5], []))
    print(multidirectional_broadcasting([2, 3, 4, 5], [5,]))
    print(multidirectional_broadcasting([4, 5], [2, 3, 4, 5,]))
    print(multidirectional_broadcasting([1, 4, 5], [2, 3, 1, 1,]))
    print(multidirectional_broadcasting([3, 4, 5], [2, 1, 1, 1,]))