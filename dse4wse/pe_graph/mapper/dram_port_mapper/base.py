from abc import ABC, abstractmethod
import numpy as np

class BaseDramPortMapper(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, virtual_dram_port_id: int):
        return (np.inf, np.inf)  # fallback to invalid point on error