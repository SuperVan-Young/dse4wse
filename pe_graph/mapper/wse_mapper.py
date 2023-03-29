
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

from typing import Tuple, List
Coordinate = Tuple[int, int]
RoutingPath = List[Coordinate]

from hardware import WaferScaleEngine
from task import BaseWaferTask

from reticle_mapper import BaseReticleMapper, XYReticleMapper
from dram_port_mapper import BaseDramPortMapper, HashDramPortMapper
from reticle_router import BaseReticleRouter, XYReticleRouter

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
        'dram_stacking_type': wse.dram_stacking_type,
    }
    reticle_router = XYReticleRouter(**reticle_router_config)

    wse_mapper = WseMapper(
        reticle_mapper=reticle_mapper,
        dram_port_mapper=dram_port_mapper,
        reticle_router=reticle_router,
    )
    return wse_mapper