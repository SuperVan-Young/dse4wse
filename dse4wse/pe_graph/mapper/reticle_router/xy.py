
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .base import BaseReticleRouter
from typing import List, Tuple
from dse4wse.utils import logger

Coordinate = Tuple[int, int]

class XYReticleRouter(BaseReticleRouter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, src: Coordinate, dst: Coordinate) -> List[Coordinate]:
        path = []
        x1, y1 = src
        x2, y2 = dst

        x_path, y_path = [], []
        for x in range(x1, x2 + (1 if x2 > x1 else -1), 1 if x2 > x1 else -1): x_path.append((x, y1))
        for y in range(y1, y2 + (1 if y2 > y1 else -1), 1 if y2 > y1 else -1): y_path.append((x2, y))

        path = x_path[:-1] + y_path
        return path

