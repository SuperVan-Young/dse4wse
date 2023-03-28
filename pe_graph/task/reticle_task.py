import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abc import ABC, abstractmethod

class BaseReticleTask():
    def __init__(self, virtual_reticle_id: int):
        self.virtual_reticle_id = virtual_reticle_id 

    @property
    @abstractmethod
    def task_type(self):
        return "base_reticle_task"
    
class ComputeReticleTask(BaseReticleTask):
    def __init__(self, virtual_reticle_id: int, latency: int) -> None:
        super().__init__(virtual_reticle_id)
        self.latency = latency

    @property
    def task_type(self):
        return 'compute'
    

class DramAccessReticleTask(BaseReticleTask):
    def __init__(self, virtual_reticle_id: int, virtual_dram_port: int, access_type: str, data_amount: int):
        super().__init__(virtual_reticle_id)
        self.virtual_dram_port = virtual_dram_port
        self.access_type = access_type
        self.data_amount = data_amount

    @property
    def task_type(self):
        return 'dram_access'
    
class PeerAccessReticleTask(BaseReticleTask):
    def __init__(self, virtual_reticle_id: int, peer_virtual_reticle_id: int, access_type: str, data_amount: int):
        super().__init__(virtual_reticle_id)
        self.peer_virtual_reticle_id = peer_virtual_reticle_id
        self.access_type = access_type
        self.data_amount = data_amount

    @property
    def task_type(self):
        return 'peer_access'
