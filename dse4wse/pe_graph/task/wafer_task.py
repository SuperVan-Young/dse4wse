import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abc import ABC
from typing import List
from reticle_task import BaseReticleTask

class BaseWaferTask(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

class ListWaferTask(BaseWaferTask):
    def __init__(self, tasklist: List[BaseReticleTask], **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = tasklist

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self) -> BaseReticleTask:
        if self.index >= len(self.data):
            raise StopIteration
        value = self.data[self.index]
        self.index += 1
        return value
    
    def __len__(self) -> int:
        return len(self.data)
    
    def append(self, value: BaseReticleTask):
        assert isinstance(value, BaseReticleTask)
        self.data.append(value)

    def get_all_virtual_reticle_ids(self) -> List[int]:
        return sorted(list(set([task.virtual_reticle_id for task in self.data])))
