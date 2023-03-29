
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Tuple, List
Coordinate = Tuple[int, int]
RoutingPath = List[Coordinate]

class WseMapper():

    def __init__(self, 
                 reticle_mapper_type: str,
                 dram_port_mapper_type: str,
                 reticle_router_type: str,
                 ) -> None:
        self._reticle_mapper = self.__init_reticle_mapper(reticle_mapper_type)
        self._dram_port_mapper = self.__init_dram_port_mapper(dram_port_mapper_type)
        self._reticle_router = self.__init_reticle_router(reticle_router_type)

    def __init_reticle_mapper(self, reticle_mapper_type: str):
        # TODO:
        raise NotImplementedError
    
    def __init_dram_port_mapper(self, dram_port_mapper_type: str):
        # TODO:
        raise NotImplementedError
    
    def __init_reticle_router(self, reticle_router_type: str):
        # TODO:
        raise NotImplementedError
    
    def find_physical_reticle_coordinate(self, virtual_reticle_id: int) -> Coordinate:
        return self._reticle_mapper(virtual_reticle_id)
    
    def find_physical_dram_port_coordinate(self, virtual_dram_port_id: int) -> Coordinate:
        return self._dram_port_mapper(virtual_dram_port_id)
    
    def find_inter_reticle_routing_path(self, src: Coordinate, dst: Coordinate, src_type: str, dst_type: str) -> RoutingPath:
        assert src_type in ['reticle', 'dram_port']
        assert dst_type in ['reticle', 'dram_port']
        return self._reticle_router(src, dst, src_type, dst_type)