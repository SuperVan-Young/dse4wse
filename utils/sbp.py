
from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import re

from copy import deepcopy
from math import sqrt
from functools import reduce
from itertools import permutations
from logger import logger
from arch_config import ArchConfig
from tensor_info import TensorInfo

SPLIT_SBP_PARALLEL = 1
BROADCAST_SBP_PARALLEL = 2
PARTIAL_SBP_PARALLEL = 3

class SbpParallel():
    def __init__(self) -> None:
        pass

    @property
    def type(self):
        return None
    
    def is_split(self) -> bool:
        return self.type == SPLIT_SBP_PARALLEL
    
    def is_broadcast(self) -> bool:
        return self.type == BROADCAST_SBP_PARALLEL
    
    def is_partial(self) -> bool:
        return self.type == PARTIAL_SBP_PARALLEL

class SplitSbpParallel(SbpParallel):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
    
    @property
    def type(self):
        return SPLIT_SBP_PARALLEL

    def __repr__(self) -> str:
        return f"S({self.dim})"

class BroadcastSbpParallel(SbpParallel):
    def __init__(self) -> None:
        super().__init__()

    @property
    def type(self):
        return BROADCAST_SBP_PARALLEL

    def __repr__(self) -> str:
        return "B"

class PartialSbpParallel(SbpParallel):
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def type(self):
        return PARTIAL_SBP_PARALLEL

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
        return SplitSbpParallel(dim=int(dim))
    else:
        raise RuntimeError("Invalid sbp parallel str %s" % (s,))
    
class Placement():
    def __init__(self, shape: Tuple[int], interconnect_types: Tuple[int] = None) -> None:
        self.shape = shape
        if interconnect_types == None:
            interconnect_types = ['core'] * len(shape)  # TODO: leave it for the future
        self.interconnect_types = interconnect_types

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Placement):
            return False
        same_shape = tuple(self.shape) == tuple(__value.shape)
        same_interconnect_types = tuple(self.interconnect_types) == tuple(__value.interconnect_types)
        return same_shape and same_interconnect_types

class SbpSignature():
    """Description of the distribution of a tensor on local devices.

    Attributes:
        - placement: local device array, the element of which points to a local VIRTUAL PE,
          i.e. a number in [0, size - 1]
        - sbp_parallel: SBP vector
    """

    def __init__(self, placement: Placement, sbp_parallels: List[SbpParallel]) -> None:
        self.placement = placement
        self.sbp_parallels = sbp_parallels
        assert len(placement.shape) == len(sbp_parallels)

    def get_total_cores(self):
        return reduce(lambda x, y: x * y, self.placement.shape)
    
    def _get_sbp_size(self, sbp_type: int):
        total_size = 1
        for dim_size, sbp_parallel in zip(self.placement.shape, self.sbp_parallels):
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
        main_str = ", ".join([f"{str(s)}: {p}" for p, s in zip(self.placement.shape, self.sbp_parallels)])
        main_str = "[" + main_str + "]"
        return main_str

def get_local_tensor_info(tensor_info: TensorInfo, sbp_signature: SbpSignature):
    tensor_info_ = deepcopy(tensor_info)
    tensor_info_.shape = list(tensor_info_.shape)
    for placement_value, sbp_parallel in zip(sbp_signature.placement.shape, sbp_signature.sbp_parallels):
        if isinstance(sbp_parallel, SplitSbpParallel):
            dim = sbp_parallel.dim
            dim_value = tensor_info_.shape[dim]
            assert dim_value % placement_value == 0
            tensor_info_.shape[dim] //= placement_value
    tensor_info_.shape = tuple(tensor_info_.shape)
    return tensor_info_

def derive_output_sbp_signatures(input_sbp_signatures: Dict[str, SbpSignature], rule_table: pd.DataFrame) -> Dict[str, SbpSignature]:
    # validate placement shape
    placements = [sbp.placement for sbp in input_sbp_signatures.values()]
    first_placement = placements[0]
    for placement_ in placements:
        if first_placement.shape != placement_.shape:
            raise ValueError("Unmatched placement when deriving sbp signatures!")
    
    lookup = {name: sbp.get_simplified_sbp_parallel_list() for name, sbp in input_sbp_signatures.items()}
    lookup = pd.DataFrame(lookup)
    lookup_result = pd.merge(left=lookup, right=rule_table, how='left')  # nan occurs on invalid derivation

    output_tensor_names = {name for name in rule_table.columns if name not in input_sbp_signatures.items()}
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

def get_grad_sbp_signature(sbp_signature: SbpSignature) -> SbpSignature:
    out_sbp_sig = deepcopy(sbp_signature)
    def change_sbp_parallel(sbp_parallel: SbpParallel):
        if sbp_parallel.is_split():
            return sbp_parallel
        elif sbp_parallel.is_broadcast():
            return PartialSbpParallel()
        elif sbp_parallel.is_partial():
            return BroadcastSbpParallel()
    out_sbp_sig.sbp_parallels = [change_sbp_parallel(sbp_prl) for sbp_prl in out_sbp_sig.sbp_parallels]
    return out_sbp_sig

def calc_comm_cost_on_same_devices(tensor_info: TensorInfo, prev_sbp_sig: SbpSignature, cur_sbp_sig: SbpSignature, arch_config: ArchConfig) -> float:
    """Estimate communication cost of rearranging global tensor on the same device, without allocating more memory
    """
    if tensor_info.inplace:
        return 0
    else:
        assert prev_sbp_sig is not None
        assert cur_sbp_sig is not None
    assert prev_sbp_sig.placement.shape == cur_sbp_sig.placement.shape
    placement = prev_sbp_sig.placement

    # current SBP of virtual transform dimensions
    # It uses inter-connection to give an abstraction of a large, uniform memory to the consumer
    # without allocating more static memory.
    # This only includes S->B
    virtual_transform_dims = []

    global_tensor_size = tensor_info.size()
    static_comm_cost = 0  # move once and for all

    for dim, (prev_parallel, cur_parallel, dim_value, interconnect_type) \
        in enumerate(zip(prev_sbp_sig.sbp_parallels, cur_sbp_sig.sbp_parallels, placement.shape, placement.interconnect_types)):
        bandwidth = arch_config.get_interconnect_bandwidth(interconnect_type)
        if prev_parallel.is_split():
            if cur_parallel.is_split():
                if prev_parallel.dim != cur_parallel.dim:
                    static_comm_cost += int(dim_value - 1)  * int(global_tensor_size / dim_value / bandwidth)
            elif cur_parallel.is_broadcast():
                virtual_transform_dims.append(dim)
            elif cur_parallel.is_partial():
                raise NotImplementedError("Seriously? Split to partial and waste your memory?")
            else:
                assert False
        elif prev_parallel.is_broadcast():
            continue
        elif prev_parallel.is_partial():
            # We assume output-stationary dataflow, so mem must be allocated for partial.
            # Neglecting input-stationary dataflow is fine, if only considering mem limitation on output tensor.
            # There're other sbp signatures that only store single copy of output tensor
            if cur_parallel.is_split():
                static_comm_cost += int(dim_value - 1)  * int(global_tensor_size / dim_value / bandwidth)
            elif cur_parallel.is_broadcast():
                static_comm_cost += 2 * int(dim_value - 1)  * int(global_tensor_size / dim_value / bandwidth)
            else:
                continue
        else:
            assert False

    # Handle virtual transform
    local_tensor_shape = list(get_local_tensor_info(tensor_info, cur_sbp_sig).shape)
    for dim in virtual_transform_dims:
        dim_value = placement.shape[dim]
        assert local_tensor_shape[dim] % dim_value == 0
        local_tensor_shape[dim] //= dim_value
    local_tensor_size = reduce(lambda x, y: x * y, local_tensor_shape)

    def calc_nested_loop_transmission_cost(info_array_):
        iterations, bandwidths = zip(*info_array_)
        iterations = np.cumprod(iterations)
        bandwidths = np.array(bandwidths)
        return np.sum(iterations / bandwidths)

    info_array = zip(
        [placement.shape[dim] for dim in virtual_transform_dims],
        [arch_config.get_interconnect_bandwidth(placement.interconnect_types[dim]) for dim in virtual_transform_dims])
    if len(virtual_transform_dims):
        minimum_virtual_transform_cost = local_tensor_size * min(
            calc_nested_loop_transmission_cost(info_array_)
            for info_array_ in permutations(info_array, len(virtual_transform_dims))
        )
    else:
        minimum_virtual_transform_cost = 0

    total_cost = static_comm_cost + minimum_virtual_transform_cost

    return total_cost


def calc_comm_cost_on_disjoint_devices():
    raise NotImplementedError

def calc_comm_cost_for_input(input_sbp_signature: Union[None, SbpSignature], output_sbp_signatures: SbpSignature, arch_config: ArchConfig, tensor_info: TensorInfo) -> float:
    """Calculate communication cost for inter layer transmission.
    """
    logger.warn("Deprecated in the future")
    # check partial dimensions
    # TODO: We could allow transmitting partial tensors with 1-to-1 routing.
    # This is beneficial for reducing unnecessary collective comms.
    # But we leave it for future implementation
    if input_sbp_signature:
        input_partial_size = input_sbp_signature.get_partial_size()
        assert input_partial_size == 1, "Currently we don't support inter-layer partial transmission"
    
    # The amount of inter-layer transmission is essentially one copy of the tensor
    # i.e. (S0, S1) -> (S0', S1')
    # We arrange both core arrays as square arrays, and take the bandwidth of intersected edge as ideal bandwidth
    # This is theoretically the BOTTLENECK, since each noc channel (both intra-array and inter-array) 
    # need to send ~ 1/sqrt(N) share of the whole tensor expectedly.
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
    if output_broadcast_size > 1:
        intra_layer_comm_cost = (sqrt(output_broadcast_size) * tensor_size) // split_cluster_bandwidth
    else:
        intra_layer_comm_cost = 0

    comm_cost = inter_layer_comm_cost + intra_layer_comm_cost

    return comm_cost


def calc_comm_cost_for_reduction(input_sbp_signature: SbpSignature, output_sbp_signature: SbpSignature, arch_config: ArchConfig, tensor_info: TensorInfo) -> float:
    """Calculate communication cost for reducing partial to split/broadcast.
    """
    logger.warn("Deprecated in the future")
    # check placement consistency
    input_placement = input_sbp_signature.placement
    output_placement = output_sbp_signature.placement
    assert input_placement.shape == output_placement.shape, "Input and output have different placement!"

    # Currently, we only adopt P -> B, and leave P -> S for future implementation.
    for input_sbp_parallel, output_sbp_parallel in zip(
        input_sbp_signature.get_simplified_sbp_parallel_list(), 
        output_sbp_signature.get_simplified_sbp_parallel_list()):
        if input_sbp_parallel != output_sbp_parallel:
            assert input_sbp_parallel == 'P' and output_sbp_parallel == "B"

    # check partial dimensions number
    partial_input_sbp_parallels = [x for x in input_sbp_signature.sbp_parallels if x.type == PARTIAL_SBP_PARALLEL]
    num_partial_dims = len(partial_input_sbp_parallels)
    assert num_partial_dims <= 1, "Currently we don't support >= 1 partial sum dimension."
    if num_partial_dims == 0:
        return 0
    
    split_size = input_sbp_signature.get_split_size()
    broadcast_size = input_sbp_signature.get_broadcast_size()
    partial_size = input_sbp_signature.get_partial_size()

    num_cluster = partial_size
    cluster_size = broadcast_size * split_size

    # The amount of transmission within a group of devices can be pre-determined.
    # The cost is 2(p - 1) * |T|, 
    # where |T| is the amount of partial sum on one device cluster.
    # p is the number of device clusters.
    # We collectively consider all other non-partial dimensions
    total_transmission = 2 * (num_cluster - 1) * (tensor_info.numel() * tensor_info.dtype_size * broadcast_size)

    # The bandwidth cannot be properly determined before placement & routing
    # For ring-based all-reduce, total_bandwidth = p * inter_cluster_bandwidth.
    # Inter_cluster_bandwidth is expected to be sqrt(#Cluster_size) * #NoC_bandwidth
    noc_bandwidth = arch_config.get_interconnect_bandwidth()
    total_bandwidth = num_cluster * sqrt(cluster_size) * noc_bandwidth

    return total_transmission // total_bandwidth