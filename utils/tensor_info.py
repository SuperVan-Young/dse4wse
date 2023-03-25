
from typing import Tuple
import re
import numpy as np
from functools import reduce
from onnx.mapping import TENSOR_TYPE_MAP

def onnx_dtype_2_storage_size(dtype: int) -> int:
    name = TENSOR_TYPE_MAP[dtype].name
    size_pattern = re.compile(r"^TensorProto\.[A-Z]+(\d+)$")

    size_match = size_pattern.match(name)
    if size_match:
        size = int(size_match.group(1)) // 8
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
    def __init__(self, name: str, shape: Tuple, onnx_dtype: int, kind: str, inplace=False) -> None:
        self.name = name
        self.shape = tuple(shape)
        self.onnx_dtype = onnx_dtype
        self.kind = kind
        self.inplace = inplace  # comm cost on same device for inplace tensor is always 0

        assert kind in ['weight', 'input', 'output', 'activation', 'constant']

    @property
    def dtype_size(self):
        return onnx_dtype_2_storage_size(self.onnx_dtype)
    
    def numel(self):
        return reduce(lambda x, y: x * y, self.shape, 1)  # shape could be empty for scalar value
    
    def size(self):
        """In Byte
        """
        return self.numel() * self.dtype_size
    
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

def transpose(shape: Tuple[int], dim0, dim1) -> Tuple[int]:
    shape = list(shape)
    tmp = shape[dim0]
    shape[dim0] = shape[dim1]
    shape[dim1] = tmp
    return tuple(shape)

if __name__ == "__main__":
    print(onnx_dtype_2_storage_size(0))

    print(multidirectional_broadcasting([2, 3, 4, 5], []))
    print(multidirectional_broadcasting([2, 3, 4, 5], [5,]))
    print(multidirectional_broadcasting([4, 5], [2, 3, 4, 5,]))
    print(multidirectional_broadcasting([1, 4, 5], [2, 3, 1, 1,]))
    print(multidirectional_broadcasting([3, 4, 5], [2, 1, 1, 1,]))