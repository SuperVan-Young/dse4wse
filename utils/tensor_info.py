
from typing import Tuple
import re
import numpy as np
from functools import reduce
from onnx.mapping import TENSOR_TYPE_MAP

def onnx_dtype_2_storage_size(dtype: int) -> int:
    name = TENSOR_TYPE_MAP[dtype].name
    size_pattern = re.compile(r"^TensorProto\.[A-Z](\d+)$")

    size_match = size_pattern.match(name)
    if size_match:
        size = size_match.group(1) // 4
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
    def __init__(self, shape: Tuple, onnx_dtype: int, name: str, inplace=False) -> None:
        self.shape = shape
        self.onnx_dtype = onnx_dtype
        self.name = name
        self.inplace = inplace

    @property
    def dtype_size(self):
        return onnx_dtype_2_storage_size(self.onnx_dtype)
    
    def numel(self):
        return reduce(lambda x, y: x * y, self.shape)