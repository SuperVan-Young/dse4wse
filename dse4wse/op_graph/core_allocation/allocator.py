import os
import sys
from typing import Dict, Union
from copy import deepcopy
from itertools import chain
from functools import reduce
from interval import interval

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph import OpGraph
from op import BaseOperator, UnaryElementwiseOperator, BinaryElementwiseOperator, MatMulOperator
from utils import logger, ArchConfig

class CoreAllocator():
    def __init__(self, arch_config: ArchConfig, num_wafer=1) -> None:
        self.arch_config = arch_config
        self.num_wafer = num_wafer

    def allocate(self, op_graph: OpGraph) -> OpGraph:
        """Allocate core range to operators in op graph.
        Parameters
            - duplication_table: the operator has multiple duplication that need to allocate cores
        """
        duplication_table = op_graph.duplication_table

        total_cores = self.arch_config.get_total_cores() * self.num_wafer
        core_memory_size = self.arch_config.get_memory_size()
        get_max_alloc_core = lambda num_core_range: max([x.sup for x in num_core_range.extrema])

        mem_total_cores = total_cores
        op_name_2_mem_core_range = self._allocate_on_memory(op_graph, mem_total_cores, core_memory_size)

        comp_total_cores = total_cores - reduce(lambda x, y: x + y, [get_max_alloc_core(core_range) * duplication_table[name] for name, core_range in op_name_2_mem_core_range.items()])
        op_name_2_comp_core_range = self._allocate_on_compute(op_graph, comp_total_cores)

        op_name_2_final_core_range = {name: op_name_2_mem_core_range[name] + op_name_2_comp_core_range[name] for name in op_graph.nodes()}

        op_graph_ = deepcopy(op_graph)
        for name, op in op_graph_.nodes(data='operator'):
            op: BaseOperator
            op.num_core_range = op_name_2_final_core_range[name]

        total_alloc_cores = int(reduce(lambda x, y: x + y, [get_max_alloc_core(op.num_core_range) * duplication_table[name] for name, op in op_graph_.nodes(data='operator')]))

        logger.info(f"Complete core allocation for OP graph {op_graph.name}")
        logger.info(f"Total cores: {total_alloc_cores} / {total_cores}")
        assert total_alloc_cores < total_cores
        return op_graph_

    def _allocate_on_memory(self, op_graph: OpGraph, total_cores: int, core_memory_size: int) -> Dict[str, interval]:
        """Allocate minimum cores based on the operator's memory utilization
        """
        duplication_table = op_graph.duplication_table
        total_memory = total_cores * core_memory_size

        op_name_2_mem_utilization = {name: op.get_mem_utilization() for name, op in op_graph.nodes(data='operator')}
        total_mem_utilization = reduce(lambda x, y: x + y, [mem * duplication_table[name] for name, mem in op_name_2_mem_utilization.items()])

        if total_mem_utilization > total_memory:
            raise RuntimeError("WSE memory used up")

        op_name_2_min_alloc_core = {name: max(int(mem_util / core_memory_size), 1) for name, mem_util in op_name_2_mem_utilization.items()}

        # allocate remaining cores to operators who requires it
        total_cores -= reduce(lambda x, y: x + y, [core * duplication_table[name] for name, core in op_name_2_min_alloc_core.items()])

        def require_extra_core(op: BaseOperator) -> int:
            factor = None
            if isinstance(op, UnaryElementwiseOperator):
                factor = 2 - 1
            elif isinstance(op, BinaryElementwiseOperator):
                factor = 4 - 1
            elif isinstance(op, MatMulOperator):
                factor = 16 - 1
            return factor
        
        op_name_2_extra_alloc_core = {name: require_extra_core(op) * op_name_2_min_alloc_core[name] 
                                      for name, op in op_graph.nodes(data='operator')}
        total_extra_cores = reduce(lambda x, y: x + y, [core * duplication_table[name] for name, core in op_name_2_extra_alloc_core.items()])
        if total_extra_cores > total_cores:
            # if there's not enough cores, everyone gets evenly smaller share
            shrink_factor = total_cores / total_extra_cores
            op_name_2_extra_alloc_core = {name: int(core * shrink_factor) for name, core in op_name_2_extra_alloc_core.items()}
        
        def get_core_range(op_name) -> interval:
            min_alloc_core = op_name_2_min_alloc_core[op_name]
            extra_alloc_core = op_name_2_extra_alloc_core[op_name]
            return interval([min_alloc_core, min_alloc_core + extra_alloc_core])
        
        op_name_2_core_range = {name: get_core_range(name) for name in op_graph.nodes()}

        return op_name_2_core_range
    
    def _allocate_on_compute(self, op_graph: OpGraph, total_cores: int) -> Dict[str, interval]:
        duplication_table = op_graph.duplication_table
        
        def require_compute_core(op: BaseOperator) -> int:
            factor = None
            if isinstance(op, UnaryElementwiseOperator):
                factor = 0
            elif isinstance(op, BinaryElementwiseOperator):
                factor = 0
            elif isinstance(op, MatMulOperator):
                factor = 1
            return factor
        
        op_name_2_comp_alloc_core_factor = {name: require_compute_core(op) for name, op in op_graph.nodes(data='operator')}
        total_comp_core_factor = reduce(lambda x, y: x + y, [factor * duplication_table[name] for name, factor in op_name_2_comp_alloc_core_factor.items()])
        op_name_2_comp_alloc_core = {name: int(total_cores * (op_name_2_comp_alloc_core_factor[name] / total_comp_core_factor)) for name in op_graph.nodes()}
        op_name_2_core_range = {name: interval([0, op_name_2_comp_alloc_core[name]]) for name in op_graph.nodes()}
        
        return op_name_2_core_range