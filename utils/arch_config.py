
from typing import Dict
from collections import UserDict
from functools import reduce
import numpy as np

class ArchConfig(UserDict):
    def __init__(self, data: Dict) -> None:
        self.data = data

    def __setitem__(self, key, item) -> None:
        raise RuntimeError("ArchConfig cannot be overwritten.")

    def __repr__(self) -> str:
        main_str = "ArchConfig {\n"
        for name, val in self.data.items():
            main_str += f"  {name}: {val}\n"
        main_str += "}\n"
        return main_str
    
    def _shallow_repr(self) -> str:
        def get_brief_name(name: str) -> str:
            brief = name.split("_")
            brief = [s[0] for s in brief]
            brief = reduce(lambda x, y: x+y, brief)
            return brief
        
        brief_config = [f"{get_brief_name(name)}{val}" for name, val in self.data.items()]
        brief_config = "_".join(brief_config)
        return brief_config
    
    def get_compute_power(self) -> int:
        """Single core computation power, in terms of Operation/Cycle.
        For simplicity, MAC on our WSE can execute one operation per cycle,
        regardless of the data type.
        """
        return self.data['core_num_mac']
    
    def get_sram_bandwidth(self) -> int:
        """Single core memory bandwidth, in terms of Byte/Cycle.
        """
        return self.data['core_buffer_width']
    
    def get_sram_size(self) -> int:
        """Single core SRAM size, in terms of Byte.
        """
        return self.data['core_buffer_size']
    
    def get_interconnect_bandwidth(self, connect_type='noc'):
        """In terms of Byte/Cycle
        Supported connect_type: noc, reticle, wafer
        """
        if connect_type == 'noc':
            return self.data['noc_bandwidth']
        elif connect_type == 'reticle':
            return self.data['inter_reticle_bandwidth']
        elif connect_type == 'wafer':
            return self.data['inter_wafer_bandwidth']
        else:
            raise NotImplementedError
        
    def get_dram_size(self) -> int:
        return np.inf
    
    def get_dram_bandwidth(self) -> int:
        """ In Bytes
        """
        #TODO: 2d-DRAM and 3d-DRAM difference?
        return self.data['dram_bandwidth']
        
    def get_total_cores(self) -> int:
        total_cores = reduce(lambda x, y: x * y, [
            self.data['reticle_array_height'],
            self.data['reticle_array_width'],
            self.data['core_array_height'],
            self.data['core_array_width'],
        ])
        return total_cores
    
    def get_reticle_array_height(self):
        return self.data['reticle_array_height']
    
    def get_reticle_array_width(self):
        return self.data['reticle_array_width']
    
if __name__ == "__main__":
    # Here's an example configuration for our WSE

    arch_config = ArchConfig({
        'core_num_mac': 32,
        'core_buffer_width': 16,
        'core_buffer_size': 256,
        'noc_virtual_channel': 4,
        'noc_buffer_size': 8,
        'noc_bandwidth': 4096,
        'core_array_height': 25,
        'core_array_width': 25,
        'reticle_array_height': 8,
        'reticle_array_width': 8,
        'inter_reticle_bandwidth': 1024,
        'inter_wafer_bandwidth': 256,
    })
    print(arch_config)
    print(arch_config._shallow_repr())