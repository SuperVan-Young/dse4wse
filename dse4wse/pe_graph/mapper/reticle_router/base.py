
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Tuple
from abc import ABC, abstractmethod

Coordinate = Tuple[int, int]

class BaseReticleRouter(ABC):
    """ return a routing path from src node to dst node.
    """

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, src: Coordinate, dst: Coordinate) -> List[Coordinate]:
        raise NotImplementedError