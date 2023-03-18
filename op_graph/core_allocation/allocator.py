import os
import sys
from copy import deepcopy
from itertools import chain
from functools import reduce

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph import OpGraph
from op import Operator
from utils import logger, ArchConfig

class CoreAllocator():
    def __init__(self, arch_config: ArchConfig, num_wafer=1) -> None:
        self.arch_config = arch_config
        self.num_wafer = num_wafer

    def allocate(self, op_graph: OpGraph) -> OpGraph:
        op_graph_ = deepcopy(op_graph)
        total_cores = self.arch_config.get_total_cores() * self.num_wafer
        total_memory = total_cores * self.arch_config.get_memory_size()
        memory_per_core = self.arch_config.get_memory_size()

        # Estimate memory consumption for each operator
        # On inference, a rough estimation is the sum of input & output tensor size
        op_name_2_memory_cost = {}
        for name, op in op_graph_.nodes(data='operator'):
            op: Operator
            memory_cost = reduce(lambda x, y: x + y, 
                                 [tensor_info.numel() * tensor_info.dtype_size for tensor_info 
                                  in chain(op.input_tensors.values(), op.output_tensors.values())])
            op_name_2_memory_cost[name] = memory_cost
        required_memory = reduce(lambda x, y: x + y, list(op_name_2_memory_cost.values()))
        if required_memory > total_memory:
            raise RuntimeError("WSE memory consumption exceeded.")
        
        # Allocation based on memory
        # Lower bound: minimum memory requirement
        # Upper bound: <= 4 * minimum memory requirement
        op_name_2_memory_core_alloc = {
            op_name: memory_cost // memory_per_core
            for op_name, memory_cost in op_name_2_memory_cost.items()
        }
        memory_core_alloc_factor = min(total_memory / required_memory, 4)

        cur_alloc_cores = reduce(lambda x, y: x + y, op_name_2_memory_core_alloc.values()) * memory_core_alloc_factor
        
        # Allocation based on computation
        left_cores = total_cores - cur_alloc_cores

        