
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict

class Core():
    def __init__(self,
                 core_num_mac: int
                 ) -> None:
        self.core_num_mac = core_num_mac

    @classmethod
    def get_compute_power(cls, core_config: Dict) -> int:
        return core_config.get('core_num_mac')