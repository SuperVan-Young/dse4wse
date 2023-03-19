import os
import sys
from typing import Dict
from copy import deepcopy
from itertools import chain
from functools import reduce
from interval import interval

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph import OpGraph
from op import Operator, UnaryElementwiseOperator, BinaryElementwiseOperator, MatMulOperator
from utils import logger, ArchConfig

class CoreAllocator():
    def __init__(self, arch_config: ArchConfig, num_wafer=1) -> None:
        self.arch_config = arch_config
        self.num_wafer = num_wafer

    def allocate(self, op_graph: OpGraph) -> OpGraph:
        total_cores = self.arch_config.get_total_cores() * self.num_wafer
        core_memory_size = self.arch_config.get_memory_size()

        op_name_2_mem_core_range = self._allocate_on_memory(op_graph, total_cores, core_memory_size)

        total_cores -= reduce(lambda x, y: x + y, [core_range[0].sup for core_range in op_name_2_mem_core_range.values()])
        op_name_2_comp_core_range = self._allocate_on_compute(op_graph, total_cores)

        op_name_2_final_core_range = {name: op_name_2_mem_core_range[name] + op_name_2_comp_core_range[name] for name in op_graph.nodes()}

        op_graph_ = deepcopy(op_graph)
        for name, op in op_graph_.nodes(data='operator'):
            op: Operator
            op.num_core_range = op_name_2_final_core_range[name]
        return op_graph_

    def _allocate_on_memory(self, op_graph: OpGraph, total_cores: int, core_memory_size: int) -> Dict[str, interval]:
        """Allocate minimum cores based on the operator's memory utilization
        """
        total_memory = total_cores * core_memory_size

        op_name_2_mem_utilization = {name: op.get_mem_utilization() for name, op in op_graph.nodes(data='operator')}
        total_mem_utilization = reduce(lambda x, y: x + y, op_name_2_mem_utilization.values())

        if total_mem_utilization > total_memory:
            raise RuntimeError("WSE memory used up")

        op_name_2_min_alloc_core = {name: int(mem_util / core_memory_size) for name, mem_util in op_name_2_mem_utilization.items()}

        # allocate remaining cores to operators who requires it
        total_cores -= reduce(lambda x, y: x + y, op_name_2_min_alloc_core.values())

        def require_extra_core(op: Operator) -> int:
            factor = None
            if isinstance(op, UnaryElementwiseOperator):
                factor = 0
            elif isinstance(op, BinaryElementwiseOperator):
                factor = 4 - 1
            elif isinstance(op, MatMulOperator):
                factor = 16 - 1
            return factor
        
        op_name_2_extra_alloc_core = {name: require_extra_core(op) * op_name_2_min_alloc_core[name] 
                                      for name, op in op_graph.nodes(data='operator')}
        total_extra_cores = reduce(lambda x, y: x + y, op_name_2_extra_alloc_core.values())
        if total_extra_cores > total_cores:
            # if there's not enough cores, everyone gets evenly smaller share
            shrink_factor = total_cores / total_extra_cores
            op_name_2_extra_alloc_core = {name: int(core * shrink_factor) for name, core in total_extra_cores}
        
        def get_core_range(op_name) -> interval:
            min_alloc_core = op_name_2_min_alloc_core[op_name]
            extra_alloc_core = op_name_2_extra_alloc_core[op_name]
            return interval([min_alloc_core, min_alloc_core + extra_alloc_core])
        
        op_name_2_core_range = {name: get_core_range(name) for name in op_graph.nodes()}

        return op_name_2_core_range
    
    def _allocate_on_compute(self, op_graph: OpGraph, total_cores: int) -> Dict[str, interval]:
        def require_compute_core(op: Operator) -> int:
            factor = None
            if isinstance(op, UnaryElementwiseOperator):
                factor = 0
            elif isinstance(op, BinaryElementwiseOperator):
                factor = 0
            elif isinstance(op, MatMulOperator):
                factor = 1
            return factor
        
        op_name_2_comp_alloc_core_factor = {name: require_compute_core(op) for name, op in op_graph.nodes(data='operator')}
        total_comp_core_factor = reduce(lambda x, y: x + y, op_name_2_comp_alloc_core_factor.values())
        op_name_2_comp_alloc_core = {name: int(total_cores * (op_name_2_comp_alloc_core_factor[name] / total_comp_core_factor)) for name in op_graph.nodes()}
        op_name_2_core_range = {name: interval([0, op_name_2_comp_alloc_core[name]]) for name in op_graph.nodes()}
        
        return op_name_2_core_range