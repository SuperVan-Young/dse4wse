
from typing import Dict
from collections import UserDict
from functools import reduce
import numpy as np

class ArchConfig(UserDict):
    """Architecture description of wafer-scale engines

    Parameters:
    - Core level:
        - core_num_mac: number of MAC units
        - core_sram_size: local SRAM size (bytes)
        - core_sram_bandwidth: local SRAM bandwidth (bytes/cycle)
        - inter_core_bandwidth: NoC bandwidth between cores
    - Reticle level:
        - core_array_height
        - core_array_width
        - inter_reticle_bandwidth
    - Wafer level:
        - reticle_array_height
        - reticle_array_width
        - wafer_dram_size: total dram size of a wafer
        - wafer_dram_bandwidth: 1 share corresponding to a reticle boundary
        - wafer_dram_stacking_type: 2d, 3d, None
        - inter_wafer_bandwidth
    """
    
    def __init__(self, data: Dict) -> None:
        self.data = data
        for key in [
            'core_num_mac',
            'core_sram_size',
            'core_sram_bandwidth',
            'inter_core_bandwidth',
            'core_array_height',
            'core_array_width',
            'inter_reticle_bandwidth',
            'reticle_array_height',
            'reticle_array_width',
            'wafer_dram_size',
            'wafer_dram_bandwidth',
            'wafer_dram_stacking_type',
            'inter_wafer_bandwidth',
        ]:
            assert key in self.data

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
        return self.data['core_sram_bandwidth']
    
    def get_sram_size(self) -> int:
        """Single core SRAM size, in terms of Byte.
        """
        return self.data['core_sram_size']
    
    def get_interconnect_bandwidth(self, connect_type='core'):
        """In terms of Byte/Cycle
        Supported connect_type: core, reticle, wafer
        """
        if connect_type == 'core':
            return self.data['inter_core_bandwidth']
        elif connect_type == 'reticle':
            return self.data['inter_reticle_bandwidth']
        elif connect_type == 'wafer':
            return self.data['inter_wafer_bandwidth']
        else:
            raise NotImplementedError
        
    def get_array_size(self, dimension):
        if dimension == 'reticle_height':
            return self.data['reticle_array_height']
        elif dimension == 'reticle_width':
            return self.data['reticle_array_width']
        elif dimension == 'core_height':
            return self.data['core_array_height']
        elif dimension == 'core_width':
            return self.data['core_array_height']
        else:
            raise NotImplementedError
        
    def get_wafer_dram_size(self) -> int:
        return self.data['wafer_dram_size']
    
    def get_wafer_dram_bandwidth(self) -> int:
        """ In Bytes
        """
        wafer_dram_stacking_type = self.data['wafer_dram_stacking_type']
        if wafer_dram_stacking_type == '2d':
            return 2 * (self.data['reticle_array_height'] + self.data['reticle_array_width']) * self.data['wafer_dram_bandwidth']
        elif wafer_dram_stacking_type == '3d':
            return self.data['reticle_array_height'] * self.data['reticle_array_width'] * self.data['wafer_dram_bandwidth']
        elif wafer_dram_stacking_type == 'none':
            return 0
        else:
            raise NotImplementedError
        
    def get_total_cores(self) -> int:
        total_cores = reduce(lambda x, y: x * y, [
            self.data['reticle_array_height'],
            self.data['reticle_array_width'],
            self.data['core_array_height'],
            self.data['core_array_width'],
        ])
        return total_cores