import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_graph.graph_constructor import BertOpGraphConstructor
from op_graph.core_allocation import CoreAllocator
from op_graph.sbp_constructor import LocalSbpConstructor
from utils import ArchConfig

# Cerebras WSE
arch_config = ArchConfig({
    'core_num_mac': 4,
    'core_buffer_width': 16,
    'core_buffer_size': 48 * 1024,
    'noc_bandwidth': 4,
    'core_array_height': 66,
    'core_array_width': 154,
    'reticle_array_height': 12,
    'reticle_array_width': 8,
})

graph_constructor = BertOpGraphConstructor()
op_graph = graph_constructor.build_op_graph()
duplication_table = graph_constructor.get_duplication_table(op_graph)

# Core Allocation
op_graph = CoreAllocator(arch_config).allocate(op_graph, duplication_table)
op_graph.profile_core_allocation()

# SBP Strategy
op_graph = LocalSbpConstructor(arch_config).find_best_strategy(op_graph)
op_graph.profile_final_sbp_signatures()