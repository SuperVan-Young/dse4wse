
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abc import ABC, abstractmethod
import numpy as np

class BaseReticleMapper(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, virtual_reticle_id: int):
        return (np.inf, np.inf)  # fallback to invalid point on error