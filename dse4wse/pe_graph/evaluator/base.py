
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

from abc import ABC, abstractmethod
from hardware import WaferScaleEngine
from task import BaseWaferTask
from mapper import WseMapper

class BaseWseEvaluator(ABC):

    def __init__(self,
                 hardware: WaferScaleEngine, 
                 task: BaseWaferTask,
                 mapper: WseMapper,
                 ) -> None:
        super().__init__()
        self.hardware = hardware
        self.task = task
        self.mapper = mapper

    @abstractmethod
    def get_total_latency(self) -> float:
        raise NotImplementedError