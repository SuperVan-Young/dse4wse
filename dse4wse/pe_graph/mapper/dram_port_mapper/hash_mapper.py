
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import hashlib

from .base import BaseDramPortMapper

class HashDramPortMapper(BaseDramPortMapper):
    def __init__(self, 
                 wafer_scale_engine,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.dram_port_coordinates = [node for node, ndata in wafer_scale_engine._reticle_graph.nodes(data=True) if ndata['dram_port']]
        self.hash_func = hashlib.sha256

    def __call__(self, virtual_dram_port_id: int):
        hash_object = self.hash_func(str(virtual_dram_port_id).encode())
        hash_val = int(hash_object.hexdigest(), 16)
        return self.dram_port_coordinates[hash_val % len(self.dram_port_coordinates)]