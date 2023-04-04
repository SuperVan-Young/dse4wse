
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict

class Core():
    def __init__(self,
                 compute_power: int,
                 core_sram_size: int,
                 core_sram_bandwidth: int,
                 ) -> None:
        self.compute_power = compute_power  # translated from core_num_mac

    @classmethod
    def get_compute_power(cls, core_config: Dict) -> int:
        return core_config.get('core_compute_power')
    
    @classmethod
    def get_sram_size(cls, core_config: Dict) -> int:
        return core_config.get('core_sram_size')
    
    @classmethod
    def get_sram_bandwidth(cls, core_config: Dict) -> int:
        return core_config.get('core_sram_bandwidth')