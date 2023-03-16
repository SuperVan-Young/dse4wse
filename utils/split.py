
from typing import List, Tuple, Union, Container
from math import sqrt
from itertools import product
from functools import reduce
import random
import numpy as np

from logger import logger
from sbp import SbpSignature, SplitSbpParallel

def factoring(num: int) -> List:
    factors = set()
    for i in range(1, int(sqrt(num))):
        if num % i == 0:
            factors.add(i)
            factors.add(num // i)
    factors.remove(1)
    return sorted(list(factors))

def generate_split_from_fixed_shape(tensor_shape: Tuple[int], split_dims: Union[int, Tuple], 
                                    core_range: Union[Container, None]=None
                                    ) -> List[SbpSignature]:
    if isinstance(split_dims, int):
        candidate_split_indices = [i for i, dim in enumerate(tensor_shape) if dim != 1]
        if split_dims > len(candidate_split_indices):
            logger.warn(f"Tensor shape {tensor_shape} cannot create split on {split_dims} dimensions.")
            return []
        random.shuffle(candidate_split_indices)
        split_indices = candidate_split_indices[:split_dims]
    elif isinstance(split_dims, [list, tuple]):
        for dim in split_dims:
            if tensor_shape[dim] == 1:
                logger.warn(f"Tensor shape {tensor_shape} cannot be split on dimension {dim}.")
                return []
        split_indices = split_dims

    factors = map(lambda i: factoring(tensor_shape[i]), split_indices)
    splits = [s for s in product(*list(factors))]
    if core_range:
        splits = [s for s in splits if reduce(lambda x, y: x * y, s) in core_range]
    
    def get_sbp_signature(split):
        num_cores = reduce(lambda x, y: x * y, split)
        placement = np.arange(0, num_cores).reshape(*split)
        sbp_parallels = [SplitSbpParallel(dim) for dim in split_indices]
        sbp_signature = SbpSignature(placement, sbp_parallels)
        return sbp_signature

    return [get_sbp_signature(split) for split in splits]
    

if __name__ == "__main__":
    print(generate_split_from_fixed_shape([24, 12, 12, 768], split_dims=3))