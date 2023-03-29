
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Tuple, List
Coordinate = Tuple[int, int]
RoutingPath = List[Coordinate]

from reticle_mapper import BaseReticleMapper
from dram_port_mapper import BaseDramPortMapper
from reticle_router import BaseReticleRouter

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
    
    def find_inter_reticle_routing_path(self, src: Coordinate, dst: Coordinate, src_type: str, dst_type: str) -> RoutingPath:
        assert src_type in ['reticle', 'dram_port']
        assert dst_type in ['reticle', 'dram_port']
        assert not (src_type == 'dram_port' and dst_type == 'dram_port')
        return self._reticle_router(src, dst, src_type, dst_type)