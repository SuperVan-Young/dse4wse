import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Any
from reticle_task import BaseReticleTask, ComputeReticleTask, DramAccessReticleTask, FusedReticleTask
from copy import deepcopy
import networkx as nx
from networkx import DiGraph

class ThreeStageReticleTaskGenerator():
    """ Read data from DRAM -> compute -> write data to DRAM.
    """

    def __init__(self,
                 compute_amount: int,
                 read_data_amount: List[int],
                 write_data_amount: List[int],
                 reuse_dram_port: bool = True,
                 ) -> None:
        self.compute_amount = compute_amount
        self.read_data_amount = read_data_amount
        self.write_data_amount = write_data_amount
        self.reuse_dram_port = reuse_dram_port

        self._reticle_counter = 0
        self._dram_port_counter = 0

    def __call__(self, **kwargs: Any) -> FusedReticleTask:
        virtual_reticle_id = self._reticle_counter
        self._reticle_counter += 1

        if self.reuse_dram_port:
            dram_port_counter = 0
        else:
            dram_port_counter = deepcopy(self._dram_port_counter)
            self._dram_port_counter += len(self.read_data_amount) + len(self.write_data_amount)

        task_graph = DiGraph()
        compute_task = ComputeReticleTask(virtual_reticle_id, self.compute_amount, **kwargs)
        task_graph.add_node('compute', task=compute_task)

        for i, data_amount in enumerate(self.read_data_amount):
            task_name = f"read_{i}"
            virtual_dram_port = dram_port_counter + i
            read_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, "read", data_amount, **kwargs)
            task_graph.add_node(task_name, task=read_task)
            task_graph.add_edge(task_name, 'compute')

        for i, data_amount in enumerate(self.write_data_amount):
            task_name = f"write_{i}"
            virtual_dram_port = dram_port_counter + len(self.read_data_amount) + i
            write_task = DramAccessReticleTask(virtual_reticle_id, virtual_dram_port, "write", data_amount, **kwargs)
            task_graph.add_node(task_name, task=write_task)
            task_graph.add_edge('compute', task_name)

        fused_task = FusedReticleTask(virtual_reticle_id, task_graph, **kwargs)
        return fused_task
