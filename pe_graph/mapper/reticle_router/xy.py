
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base import BaseReticleRouter

class XYReticleRouter(BaseReticleRouter):
    def __init__(self, 
                 dram_stacking_type: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.dram_stacking_type = dram_stacking_type
        assert dram_stacking_type == '2d', "Current impl only support 2d"

    def __call__(self, src, dst, src_type: str, dst_type: str):
        path = []
        x1, y1 = src
        x2, y2 = dst
        for x in range(x1, x2): path.append((x, y1))
        path.append((x2, y1))
        for y in range(y1, y2): path.append((x2, y))
        path.append((x2, y2))
        return path

