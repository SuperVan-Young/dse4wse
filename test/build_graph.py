import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_graph.graph_constructor import BertOpGraphConstructor
from op_graph.core_allocation import CoreAllocator
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

op_graph = BertOpGraphConstructor().build_op_graph()

# Core Allocation
op_graph = CoreAllocator(arch_config).allocate(op_graph)
op_graph.profile_core_allocation()