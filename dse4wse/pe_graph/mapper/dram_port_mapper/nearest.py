from .base import BaseDramPortMapper
from dse4wse.pe_graph.mapper.reticle_mapper import BaseReticleMapper
from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.pe_graph.task import ListWaferTask, DramAccessReticleTask, ComputeReticleTask
from dse4wse.utils import logger

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
        # self.profile_result()

    def __setup_mapping_table(self) -> Dict[int, Coordinate]:
        vdpid_2_prid = {}  # virtual dram port id -> physical reticle coordinate

        def add_dram_access_task(task: DramAccessReticleTask):
            vrid = task.virtual_reticle_id
            vdpid = task.virtual_dram_port
            prid = self.reticle_mapper(vrid)
            if not vdpid in vdpid_2_prid: vdpid_2_prid[vdpid] = []
            vdpid_2_prid[vdpid].append(prid)

        for reticle_task in self.task:
            if reticle_task.task_type == 'dram_access':
                add_dram_access_task(reticle_task)
            elif reticle_task.task_type == 'fused':
                for subtask in reticle_task.get_subtask_list():
                    if subtask.task_type == 'dram_access':
                        add_dram_access_task(subtask)
        
        def get_nearest_dram_port(coord_list):
            reticle_coords = np.array(coord_list)
            dram_ports = np.array(self.dram_port_coordinates)
            reticle_centroid = np.mean(reticle_coords, axis=0, keepdims=True)
            l1_distance = np.sum(np.abs(dram_ports - reticle_centroid), axis=1, keepdims=False)
            nearest_dram_port = self.dram_port_coordinates[np.argmin(l1_distance)]
            return nearest_dram_port

        mapping_table = {vdpid: get_nearest_dram_port(prid) for vdpid, prid in vdpid_2_prid.items()}
        return mapping_table

    def __call__(self, virtual_dram_port_id: int):
        return self.__mapping_table.get(virtual_dram_port_id, (np.inf, np.inf))
    
    def profile_result(self):
        logger.debug(f"Profiling dram port mapping result of {__name__}")

        def profile_task(task: DramAccessReticleTask):
            vrid =  task.virtual_reticle_id
            prid = self.reticle_mapper(vrid)
            vdpid = task.virtual_dram_port
            pdpid = self.__call__(vdpid)
            logger.debug(f"Reticle {vrid} {prid} -> Dram Port {vdpid} {pdpid}")

        for reticle_task in self.task:
            if reticle_task.task_type == 'dram_access':
                profile_task(reticle_task)
            elif reticle_task.task_type == 'fused':
                for subtask in reticle_task.get_subtask_list():
                    if subtask.task_type == 'dram_access':
                        profile_task(subtask)