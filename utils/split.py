
from typing import List, Tuple, Union, Container
from math import sqrt
from itertools import product
from functools import reduce
import numpy as np
from copy import deepcopy

from logger import logger
from tensor_info import TensorInfo
from sbp import SbpSignature, SplitSbpParallel

def factoring(num: int, upper_bound:int = None) -> List:
    """Return factors <= upper_bound
    """
    if upper_bound is None:
        upper_bound = num

    factors = set()
    for i in range(1, int(sqrt(num)) + 1):
        if num % i == 0:
            if i <= upper_bound: factors.add(i)
            if num // i <= upper_bound: factors.add(num // i)
    return sorted(list(factors))

def get_max_factor(num: int, upper_bound: int) -> int:
    """Return max factor <= upper_bound
    """
    if upper_bound < sqrt(num):
        for i in range(upper_bound, 0, -1):
            if num % i == 0:
                return i
    else:
        for i in range(num // upper_bound, int(sqrt(num)) + 1, 1):
            if num % i == 0:
                return num // i
            
def get_split_tensor_info(tensor_info: TensorInfo, sbp_signature: SbpSignature):
    tensor_info_ = deepcopy(tensor_info)
    tensor_info_.shape = list(tensor_info_.shape)
    for placement_value, sbp_parallel in zip(sbp_signature.placement.shape, sbp_signature.sbp_parallels):
        if isinstance(sbp_parallel, SplitSbpParallel):
            dim = sbp_parallel.dim
            dim_value = tensor_info_.shape[dim]
            assert dim_value % placement_value == 0
            tensor_info_.shape[dim] //= placement_value
    tensor_info_.shape = tuple(tensor_info_.shape)
    return tensor_info_