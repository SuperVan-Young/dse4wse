
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abc import ABC, abstractmethod

class BaseReticleRouter(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, src, dst, src_type, dst_type):
        """
        r2r: [r, r, ..., r]
        d2r: [d, r, ..., r]
        r2d: [r, r, ..., d]
        """
        raise NotImplementedError