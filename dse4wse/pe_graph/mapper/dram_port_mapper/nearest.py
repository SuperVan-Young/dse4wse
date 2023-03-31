from .base import BaseDramPortMapper
from dse4wse.pe_graph.mapper.reticle_mapper import BaseReticleMapper
from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.pe_graph.task import ListWaferTask, DramAccessReticleTask, ComputeReticleTask

from typing import Dict, Tuple
Coordinate = Tuple[int, int]

import numpy as np

class NearestDramPortMapper(BaseDramPortMapper):
    def __init__(self,
                wafer_scale_engine: WaferScaleEngine,
                reticle_mapper: BaseReticleMapper,
                task: ListWaferTask,
                **kwargs,
                ) -> None:
        super().__init__(**kwargs)
        self.dram_port_coordinates = [node for node, ndata in wafer_scale_engine._reticle_graph.nodes(data=True) if ndata['dram_port']]
        self.reticle_mapper = reticle_mapper
        self.task = task
        self.__mapping_table = self.__setup_mapping_table()

    def __setup_mapping_table(self) -> Dict[int, Coordinate]:
        vdpid_2_prcoord = {}  # virtual dram port id -> physical reticle coordinate

        for reticle_task in self.task:
            assert reticle_task.task_type == 'fused'
            for subtask in reticle_task.get_subtask_list():
                if subtask.task_type == 'compute':
                    continue                    
                elif subtask.task_type == 'dram_access':
                    subtask: DramAccessReticleTask
                    virtual_dram_port_id = subtask.virtual_dram_port
                    physical_reticle_coordinate = self.reticle_mapper(subtask.virtual_reticle_id)
                    if not virtual_dram_port_id in vdpid_2_prcoord: vdpid_2_prcoord[virtual_dram_port_id] = []
                    vdpid_2_prcoord[virtual_dram_port_id].append(physical_reticle_coordinate)
                else:
                    raise NotImplementedError(f"Unrecognized subtask type {subtask.type}")
        
        def get_nearest_dram_port(coord_list):
            dram_ports = np.array(coord_list)
            centroid = np.mean(dram_ports, axis=1, keepdims=True)
            l1_distance = np.sum(np.abs(dram_ports - centroid), axis=1, keepdims=False)
            nearest_dram_port = self.dram_port_coordinates[np.argmin(l1_distance)]
            return nearest_dram_port
        mapping_table = {vdpid: get_nearest_dram_port(prcoord) for vdpid, prcoord in vdpid_2_prcoord.items()}
        return mapping_table

    def __call__(self, virtual_dram_port_id: int):
        return self.__mapping_table.get(virtual_dram_port_id, (np.inf, np.inf))