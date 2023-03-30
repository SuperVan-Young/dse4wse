import os
import sys
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph import OpGraph
from op import BaseOperator
from utils import logger, ArchConfig

class LocalSbpConstructor():
    """Local SBP parallel strategy constructor
    No consideration on input sbp signature
    """
    def __init__(self, arch_config: ArchConfig) -> None:
        self.arch_config = arch_config

    def find_best_strategy(self, op_graph: OpGraph) -> OpGraph:
        op_graph_ = deepcopy(op_graph)
        for name, op in op_graph_.nodes(data='operator'):
            op: BaseOperator
            op.generate_candidate_sbp_signatures()
            logger.debug(f"operator {name} candidate sbp signatures: {len(op._candidate_sbp_signatures)}")
            final_sbp_sig = op.find_best_sbp_signature(arch_config=self.arch_config, inter_layer_sbp_signatures={})
            op.final_sbp_signatures = final_sbp_sig

            logger.info(f"Complete finding SBP strategy for operator {name}")
        return op_graph_
