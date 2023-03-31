
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict

from .core import Core

class Reticle():
    def __init__(self, 
                 core_array_height: int,
                 core_array_width: int,
                 inter_core_bandwidth: int,
                 core_config: Dict,
                 **kwargs) -> None:
        self.core_array_height = core_array_height
        self.core_array_width = core_array_width
        self.inter_core_bandwidth = inter_core_bandwidth
        self.core_config = core_config

        self.coordinate = kwargs.get('coordinate', None)

        self._core_graph = self.__build_core_graph()

    @classmethod
    def get_compute_power(cls, reticle_config: Dict) -> int:
        core_array_height = reticle_config['core_array_height']
        core_array_width = reticle_config['core_array_width']
        core_compute_power = Core.get_compute_power(reticle_config['core_config'])
        reticle_compute_power = core_array_height * core_array_width * core_compute_power
        return reticle_compute_power
    
    @classmethod
    def get_sram_size(cls, reticle_config: Dict) -> int:
        core_array_height = reticle_config['core_array_height']
        core_array_width = reticle_config['core_array_width']
        core_sram_size = Core.get_sram_size(reticle_config['core_config'])
        reticle_sram_size = core_array_height * core_array_width * core_sram_size
        return reticle_sram_size
    
    def __build_core_graph(self):
        return None