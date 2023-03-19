import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from networkx import DiGraph

from op import Operator
from utils import logger

class OpGraph(DiGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.name = None
        self._onnx_graph = None

    def profile_core_allocation(self):
        logger.info(f"Profiling core allocation of OP graph {self.name}")
        for node, op in self.nodes(data='operator'):
            op: Operator
            logger.info(f"{node}: {op.num_core_range}")
            
            upper_bound = max([x.sup for x in op.num_core_range.extrema])
            if upper_bound == 0:
                logger.warn(f"{node} doesn't have allocated cores.")

    def profile_final_sbp_signatures(self):
        logger.info(f"Profiling final SBP signatures of OP graph {self.name}")
        for node, op in self.nodes(data='operator'):
            op: Operator
            logger.info(f"{node}: {op.final_sbp_signatures}")
