
from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import re

from copy import deepcopy
from math import sqrt
from functools import reduce
from logger import logger
from arch_config import ArchConfig
from tensor_info import TensorInfo

SPLIT_SBP_PARALLEL = 1
BROADCAST_SBP_PARALLEL = 2
PARTIAL_SBP_PARALLEL = 3

class SbpParallel():
    def __init__(self) -> None:
        self.type = None

class SplitSbpParallel(SbpParallel):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.type = SPLIT_SBP_PARALLEL
        self.dim = dim

    def __repr__(self) -> str:
        return f"S({self.dim})"

class BroadcastSbpParallel(SbpParallel):
    def __init__(self) -> None:
        super().__init__()
        self.type = BROADCAST_SBP_PARALLEL

    def __repr__(self) -> str:
        return "B"

class PartialSbpParallel(SbpParallel):
    def __init__(self) -> None:
        super().__init__()
        self.type = PARTIAL_SBP_PARALLEL

    def __repr__(self) -> str:
        return "P"
    
def get_sbp_parallel_from_str(s: str) -> SbpParallel:
    assert type(s) == str

    split_pattern = re.compile(r"^S\((\d)\)$")
    split_match = split_pattern.match(s)

    if s == 'B':
        return BroadcastSbpParallel()
    elif s == 'P':
        return PartialSbpParallel()
    elif split_match:
        dim = split_match.group(1)
        return SplitSbpParallel(dim=dim)
    else:
        raise RuntimeError("Invalid sbp parallel str %s" % (s,))


class SbpSignature():
    """Description of the distribution of a tensor on local devices.

    Attributes:
        - placement: local device array, the element of which points to a local VIRTUAL PE,
          i.e. a number in [0, size - 1]
        - sbp_parallel: SBP vector
    """

    def __init__(self, placement: np.array, sbp_parallels: List[SbpParallel]) -> None:
        self.placement = placement
        self.sbp_parallels = sbp_parallels
        assert len(placement.shape) == len(sbp_parallels)

    def get_total_cores(self):
        return reduce(lambda x, y: x * y, self.placement)
    
    def _get_sbp_size(self, sbp_type: int):
        total_size = 1
        for dim_size, sbp_parallel in zip(self.placement, self.sbp_parallels):
            if sbp_parallel.type == sbp_type:
                total_size *= dim_size
        return total_size

    def get_broadcast_size(self):
        return self._get_sbp_size(BROADCAST_SBP_PARALLEL)
    
    def get_split_size(self):
        return self._get_sbp_size(SPLIT_SBP_PARALLEL)
    
    def get_partial_size(self):
        return self._get_sbp_size(PARTIAL_SBP_PARALLEL)

    def get_simplified_sbp_parallel_list(self):
        return [str(s) for s in self.sbp_parallels]
    
    def __repr__(self):
        main_str = ",".join([f"{str(s)}: {p}" for p, s in zip(self.placement.shape, self.sbp_parallels)])
        main_str = "Sbp Signature [" + main_str + "]"
        return main_str


def derive_output_sbp_signature(input_sbp_signatures: Dict[str, SbpSignature], rule_table: pd.DataFrame) -> Dict[str, SbpSignature]:
    # validate placement shape
    placements = [sbp.placement for sbp in input_sbp_signatures.values()]
    first_placement = placements[0]
    for placement_ in placements:
        if first_placement.shape != placement_.shape:
            raise ValueError("Unmatched placement when deriving sbp signatures!")
    
    lookup = {name: sbp.get_simplified_sbp_parallel_list() for name, sbp in input_sbp_signatures}
    lookup = pd.DataFrame(lookup)
    lookup_result = pd.merge(left=lookup, right=rule_table, how='left')  # nan occurs on invalid derivation

    output_tensor_names = {name for name in rule_table.columns if name not in input_sbp_signatures}
    output_sbp_signatures = {}
    for name in output_tensor_names:
        try:
            output_tensor_sbp_parallels = [get_sbp_parallel_from_str(s) for s in lookup_result[name]]
            output_tensor_sbp_signature = SbpSignature(first_placement, output_tensor_sbp_parallels)
            output_sbp_signatures[name] = output_tensor_sbp_signature
        except:
            logger.debug(input_sbp_signatures)
            raise RuntimeError("Cannot find valid derivation for current sbp signature!")
    return output_sbp_signatures


def derive_reduced_sbp_signatures(tensor_shape: Tuple, sbp_signature: SbpSignature) -> List[SbpSignature]:
    """Reduce Partial to existing Split or Broadcast in the signature.
    This should only be used for reorganizing output tensor of an operator.
    """
    #TODO: split is a little bit tricky, and will be implemented in the future
    new_sbp_signature = deepcopy(sbp_signature)
    new_sbp_signature.sbp_parallels = [
        x if not x.type == PARTIAL_SBP_PARALLEL else BroadcastSbpParallel()
          for x in new_sbp_signature.sbp_parallels 
    ]
    return [new_sbp_signature]


def calc_comm_cost_for_input(input_sbp_signature: Union[None, SbpSignature], output_sbp_signatures: SbpSignature, arch_config: ArchConfig, tensor_info: TensorInfo) -> float:
    """Calculate communication cost for inter layer transmission.
    """
    # check partial dimensions
    # TODO: We could allow transmitting partial tensors with 1-to-1 routing.
    # This is beneficial for reducing unnecessary collective comms.
    # But we leave it for future implementation
    if input_sbp_signature:
        input_partial_size = input_sbp_signature.get_partial_size()
        assert input_partial_size == 1, "Currently we don't support inter-layer partial transmission"
    
    # The amount of inter-layer transmission is essentially one copy of the tensor
    # i.e. (S0, S1) -> (S0', S1')
    # We take a coarse estimation of inter-layer bandwidth here
    output_split_size = output_sbp_signatures.get_split_size()
    if input_sbp_signature:
        input_split_size = input_sbp_signature.get_split_size()
        split_cluster_bandwidth = sqrt(min(input_split_size, output_split_size)) * arch_config.get_interconnect_bandwidth()
    else:
        split_cluster_bandwidth = sqrt(output_split_size) * arch_config.get_interconnect_bandwidth()
    tensor_size = tensor_info.numel() * tensor_info.dtype_size

    inter_layer_comm_cost = tensor_size // split_cluster_bandwidth

    # Intra-layer broadcasting
    # To form a broadcasting tree on a 2d array of N cores, the critical path is from center to corner, whose length ~ sqrt(N)
    # Each 'core' here is a split cluster, whose bandwidth has been estimated
    output_broadcast_size = output_sbp_signatures.get_broadcast_size()
    intra_layer_comm_cost = (sqrt(output_broadcast_size) * tensor_size) // split_cluster_bandwidth

    comm_cost = inter_layer_comm_cost + intra_layer_comm_cost

    return comm_cost


def calc_comm_cost_for_reduction(input_sbp_signature: SbpSignature, output_sbp_signature: SbpSignature, arch_config: ArchConfig, tensor_info: TensorInfo) -> float:
    """Calculate communication cost for reducing partial to split/broadcast.
    """
    # check placement consistency
    input_placement = input_sbp_signature.placement.reshape(-1)
    output_placement = output_sbp_signature.placement.reshape(-1)
    assert len(input_placement) == len(output_placement), "Different number of cores!"
    assert input_placement == output_placement, "Permutation on placement!"

    # check partial dimensions
    partial_input_sbp_parallels = [x for x in input_sbp_signature.sbp_parallels if x.type == PARTIAL_SBP_PARALLEL]
    num_partial_dims = len(partial_input_sbp_parallels)
    assert num_partial_dims <= 1, "Currently we don't support >= 1 partial sum dimension."
    if num_partial_dims == 0:
        return 0
    
    # calculate block size / cluster size for each dimension
    block_sizes = []
    cur_block_size = tensor_info.numel()
    for dim_size, sbp_parallel in zip(input_sbp_signature.placement.shape, input_sbp_signature.sbp_parallels):
        if sbp_parallel.type == SPLIT_SBP_PARALLEL:
            cur_block_size /= dim_size
        block_sizes.append(cur_block_size)
    
    # calculate cluster size for each dimension
    total_core = input_sbp_signature.get_total_cores()
    cluster_sizes = []
    cur_cluster_number = 1
    for dim_size in input_sbp_signature.placement.shape:
        cluster_sizes.append(total_core / cur_cluster_number)
        cur_cluster_number *= dim_size

    # The amount of transmission within a group of devices can be pre-determined.
    # Currently, we only adopt P -> B, and leave P -> S for future implementation.
    # The cost is 2(p - 1) * |T|, which assumes that block size is divisible by p.
    # This assumption usually holds true, and should work fine for coarse modeling.
    for input_sbp_parallel, output_sbp_parallel in zip(
        input_sbp_signature.get_simplified_sbp_parallel_list(), 
        output_sbp_signature.get_simplified_sbp_parallel_list()):
        if input_sbp_parallel != output_sbp_parallel:
            assert input_sbp_parallel == 'P' and output_sbp_parallel == "B"
    total_transmission = 0
    for dim_size, sbp_parallel, block_size in zip(input_sbp_signature.placement.shape, input_sbp_signature.sbp_parallels, block_sizes):
        if sbp_parallel.type == PARTIAL_SBP_PARALLEL:
            total_transmission = 2 * (dim_size - 1) * block_size * tensor_info.dtype_size

    # The bandwidth cannot be properly determined before placement & routing
    # For ring-based all-reduce, total_bandwidth = p * inter_cluster_bandwidth.
    # Inter_cluster_bandwidth is optimisitally set with #cluster_core * #NoC_bw.
    total_bandwidth = 0
    noc_bandwidth = arch_config.get_interconnect_bandwidth()
    for dim_size, sbp_parallel, cluster_size in zip(input_sbp_signature.placement.shape, input_sbp_signature.sbp_parallels, cluster_sizes):
        if sbp_parallel.type == PARTIAL_SBP_PARALLEL:
            total_bandwidth = dim_size * cluster_size * noc_bandwidth

    return total_transmission / total_bandwidth