
from typing import List, Tuple, Union, Container
from math import sqrt
from itertools import product
from functools import reduce
import random
import numpy as np

from logger import logger
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