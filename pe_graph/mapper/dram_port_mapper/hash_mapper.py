
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base import BaseDramPortMapper

class HashDramPortMapper(BaseDramPortMapper):
    def __init__(self, 
                 wafer_scale_engine,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.dram_port_coordinates = wafer_scale_engine.get_dram_port_coordinate_list()

    def __call__(self, virtual_dram_port_id: int):
        return self.dram_port_coordinates[virtual_dram_port_id % len(self.dram_port_coordinates)]