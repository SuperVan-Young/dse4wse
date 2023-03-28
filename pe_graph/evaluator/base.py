
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

from abc import ABC, abstractmethod
from hardware import WaferScaleEngine


class BaseWseEvaluator(ABC):
    
    @abstractmethod
    def get_throughput(self, wse: WaferScaleEngine) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_fp_latency(self, wse: WaferScaleEngine) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def get_bp_latency(self, wse: WaferScaleEngine) -> float:
        raise NotImplementedError