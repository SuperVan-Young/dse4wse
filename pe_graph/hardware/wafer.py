
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict
from itertools import product
import networkx as nx

from reticle import ReticleArray
from dram_port import DramPort

class WaferScaleEngine():

    def __init__(self,
                 reticle_array_height: int,
                 reticle_array_width: int,
                 inter_reticle_bandwidth: int,
                 dram_bandwidth: int,
                 dram_stacking_type: str,
                 reticle_config: Dict,
                 **kwargs,
                 ) -> None:
        
        self.reticle_array_height = reticle_array_height
        self.reticle_array_width = reticle_array_width
        self.inter_reticle_bandwidth = inter_reticle_bandwidth
        self.dram_bandwidth = dram_bandwidth
        self.dram_stacking_type = self.dram_stacking_type
        assert dram_stacking_type in ['2d', '3d']

        self._reticle_graph = self.__build_reticle_graph(reticle_config)
        self._reticle_tasks = []
        self.reticle_mapper = None    # virtual_reticle_id -> coordinate
        self.dram_port_mapper = None  # virtual_port_id -> coordinate
        self.reticle_router = None    # (src_coordinate, dst_coordinate) -> path [(u, v)]

    
    def __build_reticle_graph(self, reticle_config: Dict):
        G = nx.DiGraph()

        def add_reticle(x, y, reticle: ReticleArray):
            G.add_node(self.__reticle_naming(x, y), hw_type='reticle', reticle=reticle)

        def add_dram_port(x, y, dram_port: DramPort):
            G.add_node(self.__dram_port_naming(x, y), hw_type='dram_port', dram_port=dram_port)

        def add_reticle_2_reticle_bidirectional_link(x1, y1, x2, y2):
            G.add_edge(self.__reticle_naming(x1, y1), self.__reticle_naming(x2, y2))
            G.add_edge(self.__reticle_naming(x2, y2), self.__reticle_naming(x1, y1))
        
        def add_port_2_reticle_bidirectional_link(xp, yp, xr, yr):
            G.add_edge(self.__dram_port_naming(xp, yp), self.__reticle_naming(xr, yr))
            G.add_edge(self.__reticle_naming(xr, yr), self.__dram_port_naming(xp, yp))

        # reticle arrays
        for x, y in product(range(self.reticle_array_height), range(self.reticle_array_width)):
            reticle = ReticleArray(coordinate=(x, y), **reticle_config)
            add_reticle(x, y, reticle)
        for x, y in product(range(self.reticle_array_height - 1), range(self.reticle_array_width)):
            add_reticle_2_reticle_bidirectional_link(x-1, y, x, y)
        for x, y in product(range(self.reticle_array_height), range(self.reticle_array_width - 1)):
            add_reticle_2_reticle_bidirectional_link(x, y-1, x, y)

        # dram ports
        if self.dram_stacking_type == '2d':
            for x in range(0, self.reticle_array_height):
                add_dram_port(x, -1, DramPort())
                add_dram_port(x, self.reticle_array_width, DramPort())
                add_port_2_reticle_bidirectional_link(x, -1, x, 0)
                add_port_2_reticle_bidirectional_link(x, self.reticle_array_width, x, self.reticle_array_width-1)
            for y in range(0, self.reticle_array_width):
                add_dram_port(-1, y, DramPort())
                add_dram_port(self.reticle_array_height, y, DramPort())
                add_port_2_reticle_bidirectional_link(-1, y, 0, y)
                add_port_2_reticle_bidirectional_link(self.reticle_array_height, y, self.reticle_array_height-1, 0)
        elif self.dram_stacking_type == '3d':
            for x, y in product(range(self.reticle_array_height), range(self.reticle_array_width)):
                add_dram_port(x, y, DramPort())
                add_port_2_reticle_bidirectional_link(x, y, x, y)
        else:
            raise NotImplementedError

        return G

            
    def get_reticle_from_coordinate(self, x: int, y: int) -> ReticleArray:
        assert x >= 0 and x < self.reticle_array_height
        assert y >= 0 and y < self.reticle_array_width
        return self._reticle_graph.nodes[self.__reticle_naming(x, y)]['reticle']
    
    def get_dram_port_from_coordinate(self, x: int, y: int) -> DramPort:
        if self.dram_stacking_type == '2d':
            assert x in [-1, self.reticle_array_height] or y in [-1, self.reticle_array_width]
            return self._reticle_graph.nodes[self.__dram_port_naming(x, y)]['dram_port']
        elif self.dram_stacking_type == '3d':
            assert x in range(self.reticle_array_height) and y in range(self.reticle_array_width)
            return self._reticle_graph.nodes[self.__dram_port_naming(x, y)]['dram_port']
        else:
            raise RuntimeError

    def __reticle_naming(self, x: int, y: int) -> str:
        return f"r_{x}_{y}"
    
    def __dram_port_naming(self, x: int, y: int) -> str:
        return f"d_{x}_{y}"
