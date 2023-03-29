import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abc import ABC, abstractmethod
import networkx as nx
from networkx import DiGraph
import numpy as np

class BaseReticleTask():
    def __init__(self, virtual_reticle_id: int, **kwargs):
        self.virtual_reticle_id = virtual_reticle_id
        self.data_parallel_idx = kwargs.get('data_parallel_idx', None)
        self.tensor_parallel_idx = kwargs.get('tensor_parallel_idx', None)
        self.pipeline_parallel_idx = kwargs.get('pipeline_parallel_idx', None)

    @property
    @abstractmethod
    def task_type(self):
        return "base_reticle_task"
    
class ComputeReticleTask(BaseReticleTask):
    def __init__(self, virtual_reticle_id: int, compute_amount: int, **kwargs) -> None:
        super().__init__(virtual_reticle_id, **kwargs)
        self.compute_amount = compute_amount

    @property
    def task_type(self):
        return 'compute'

class DramAccessReticleTask(BaseReticleTask):
    def __init__(self, virtual_reticle_id: int, virtual_dram_port: int, access_type: str, data_amount: int, **kwargs):
        super().__init__(virtual_reticle_id, **kwargs)
        self.virtual_dram_port = virtual_dram_port
        self.access_type = access_type
        self.data_amount = data_amount
        assert access_type in ['read', 'write']

    @property
    def task_type(self):
        return 'dram_access'
    
class PeerAccessReticleTask(BaseReticleTask):
    def __init__(self, virtual_reticle_id: int, peer_virtual_reticle_id: int, access_type: str, data_amount: int, **kwargs):
        super().__init__(virtual_reticle_id, **kwargs)
        self.peer_virtual_reticle_id = peer_virtual_reticle_id
        self.access_type = access_type
        self.data_amount = data_amount
        assert access_type in ['read', 'write']

    @property
    def task_type(self):
        return 'peer_access'

class FusedReticleTask(BaseReticleTask):
    def __init__(self, virtual_reticle_id: int, task_graph: DiGraph, repeated_times: int = np.inf, **kwargs):
        super().__init__(virtual_reticle_id, **kwargs)
        assert nx.is_directed_acyclic_graph(task_graph)
        self.task_graph = task_graph
        self.repeated_times = repeated_times

    @property
    def task_type(self):
        return 'fused'