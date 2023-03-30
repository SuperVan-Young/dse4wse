
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base import BaseReticleRouter
from typing import List, Tuple
Coordinate = Tuple[int, int]

class XYReticleRouter(BaseReticleRouter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, src: Coordinate, dst: Coordinate) -> List[Coordinate]:
        path = []
        x1, y1 = src
        x2, y2 = dst
        for x in range(x1, x2): path.append((x, y1))
        path.append((x2, y1))
        for y in range(y1, y2): path.append((x2, y))
        path.append((x2, y2))
        return path

