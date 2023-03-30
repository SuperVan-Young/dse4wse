
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

from typing import Tuple, List
Coordinate = Tuple[int, int]
Link = Tuple[Coordinate, Coordinate]
Path = List[Coordinate]
LinkList = List[Link]

from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.pe_graph.task import BaseWaferTask

from .reticle_mapper import BaseReticleMapper, XYReticleMapper
from .dram_port_mapper import BaseDramPortMapper, HashDramPortMapper
from .reticle_router import BaseReticleRouter, XYReticleRouter

class WseMapper():
    """Leave the problem of instantiating children mappers to users
    We only provide an API to users.
    """

    def __init__(self, 
                 reticle_mapper: BaseReticleMapper,
                 dram_port_mapper: BaseDramPortMapper,
                 reticle_router: BaseReticleRouter
                 ) -> None:
        self._reticle_mapper = reticle_mapper
        self._dram_port_mapper = dram_port_mapper
        self._reticle_router = reticle_router
    
    def find_physical_reticle_coordinate(self, virtual_reticle_id: int) -> Coordinate:
        return self._reticle_mapper(virtual_reticle_id)
    
    def find_physical_dram_port_coordinate(self, virtual_dram_port_id: int) -> Coordinate:
        return self._dram_port_mapper(virtual_dram_port_id)
    
    def find_read_dram_routing_path(self, reticle_coordinate: Coordinate, dram_port_coordinate: Coordinate) -> LinkList:
        return self.__path_2_link_list(self._reticle_router(dram_port_coordinate, reticle_coordinate))
    
    def find_write_dram_routing_path(self, reticle_coordinate: Coordinate, dram_port_coordinate: Coordinate) -> LinkList:
        return self.__path_2_link_list(self._reticle_router(reticle_coordinate, dram_port_coordinate))
    
    def find_read_peer_routing_path(self, reticle_coordinate: Coordinate, peer_reticle_coordinate: Coordinate) -> LinkList:
        return self.__path_2_link_list(self._reticle_router(peer_reticle_coordinate, reticle_coordinate))
    
    def find_write_peer_routing_path(self, reticle_coordinate: Coordinate, peer_reticle_coordinate: Coordinate) -> LinkList:
        return self.__path_2_link_list(self._reticle_router(reticle_coordinate, peer_reticle_coordinate))
    
    def __path_2_link_list(self, path: Path) -> LinkList:
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]

    
def get_default_mapper(wse: WaferScaleEngine):
    reticle_mapper_config = {
        'reticle_array_height': wse.reticle_array_height,
        'reticle_array_width': wse.reticle_array_width,
    }
    reticle_mapper = XYReticleMapper(**reticle_mapper_config)

    dram_port_mapper_config = {
        'wafer_scale_engine': wse,
    }
    dram_port_mapper = HashDramPortMapper(**dram_port_mapper_config)

    reticle_router_config = {
    }
    reticle_router = XYReticleRouter(**reticle_router_config)

    wse_mapper = WseMapper(
        reticle_mapper=reticle_mapper,
        dram_port_mapper=dram_port_mapper,
        reticle_router=reticle_router,
    )
    return wse_mapper