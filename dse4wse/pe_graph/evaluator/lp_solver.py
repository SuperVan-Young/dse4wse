
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from copy import deepcopy
import networkx as nx
from networkx import DiGraph
from scipy.optimize import linprog
import numpy as np
import torch as th
from itertools import chain, product

from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.pe_graph.task import BaseReticleTask, ListWaferTask, ComputeReticleTask, DramAccessReticleTask, PeerAccessReticleTask, FusedReticleTask
from dse4wse.pe_graph.mapper import WseMapper
from dse4wse.utils import logger

from .base import BaseWseEvaluator

class LpReticleLevelWseEvaluator(BaseWseEvaluator):
    """ Use linear programming to estimate reticle-level performance.
    The overall latency is determined by the slowest reticle-level task
    """

    def __init__(self, 
                 hardware: WaferScaleEngine, 
                 task: ListWaferTask, 
                 mapper: WseMapper
                 ) -> None:
        super().__init__(hardware, task, mapper)
        self.vrid_2_var = {vrid: i for i, vrid in enumerate(task.get_all_virtual_reticle_ids())}

    def get_total_latency(self) -> float:
        G = self.__build_annotated_graph()
        min_freq = self.__lp_solver(G)  # times / second
        repeated_times = max([reticle_task.repeated_times for reticle_task in self.task])  # times
        return repeated_times / min_freq

    def __build_annotated_graph(self) -> DiGraph:
        # get annotated graph (directly use a graph copy for agile impl)
        G = deepcopy(self.hardware._reticle_graph)
        for node, ndata in G.nodes(data=True):
            ndata['compute_mark'] = {}
            ndata['dram_access_mark'] = {}
        for u, v, edata in G.edges(data=True):
            edata['transmission_mark'] = {}

        def add_compute_task(task: ComputeReticleTask):
            vrid = task.virtual_reticle_id
            prid = self.mapper.find_physical_reticle_coordinate(vrid)
            ndata = G.nodes[prid]
            if self.vrid_2_var[vrid] not in ndata['compute_mark']:
                ndata['compute_mark'][self.vrid_2_var[vrid]] = task.compute_amount
            else:
                ndata['compute_mark'][self.vrid_2_var[vrid]] += task.compute_amount

        def add_dram_access_task(task: DramAccessReticleTask):
            vrid = task.virtual_reticle_id
            prid = self.mapper.find_physical_reticle_coordinate(vrid)
            pdpid = self.mapper.find_physical_dram_port_coordinate(task.virtual_dram_port)
            ndata = G.nodes[pdpid]
            if self.vrid_2_var[vrid] not in ndata['dram_access_mark']:
                ndata['dram_access_mark'][self.vrid_2_var[vrid]] = task.data_amount
            else:
                ndata['dram_access_mark'][self.vrid_2_var[vrid]] += task.data_amount

            routing_func = self.mapper.find_read_dram_routing_path \
                           if task.access_type == 'read' else self.mapper.find_write_dram_routing_path
            link_list = routing_func(prid, pdpid)
            for link in link_list:
                edata = G.edges[link]
                if self.vrid_2_var[vrid] not in edata['transmission_mark']:
                    edata['transmission_mark'][self.vrid_2_var[vrid]] = task.data_amount
                else:
                    edata['transmission_mark'][self.vrid_2_var[vrid]] += task.data_amount

        def add_peer_access_task(task: PeerAccessReticleTask):
            vrid = task.virtual_reticle_id
            prid = self.mapper.find_physical_reticle_coordinate(vrid)
            peer_prid = self.mapper.find_physical_reticle_coordinate(task.peer_virtual_reticle_id)
            routing_func = self.mapper.find_read_peer_routing_path \
                           if task.access_type == 'read' else self.mapper.find_write_peer_routing_path
            link_list = routing_func(prid, peer_prid)
            for link in link_list:
                edata = G.edges[link]
                if self.vrid_2_var[vrid] not in edata['transmission_mark']:
                    edata['transmission_mark'][self.vrid_2_var[vrid]] = task.data_amount
                else:
                    edata['transmission_mark'][self.vrid_2_var[vrid]] += task.data_amount

        def add_task(task: BaseReticleTask):
            if isinstance(task, ComputeReticleTask):
                add_compute_task(task)
            elif isinstance(task, DramAccessReticleTask):
                add_dram_access_task(task)
            elif isinstance(task, PeerAccessReticleTask):
                add_peer_access_task(task)
            else:
                raise NotImplementedError(f"Unrecognized subtask type {task.task_type}")

        for reticle_task in self.task:
            if reticle_task.task_type == 'fused':
                reticle_task: FusedReticleTask
                for subtask in reticle_task.get_subtask_list():
                    add_task(subtask)
            else:
                add_task(reticle_task)

        return G
        
    def __lp_solver(self, G: DiGraph) -> float:
        """
        Calculate the slowest frequency of all reticle tasks            
        """
        global_freq_index = len(self.vrid_2_var)
        num_variables = len(self.vrid_2_var) + 1

        c = np.zeros(num_variables)  # f_0, f_1, ..., f_{n-1}, f
        c[global_freq_index] = -1  # maximize f
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        bounds = [(0, None) for _ in range(num_variables)]

        def add_compute_constraint(ndata):
            reticle_compute_power = self.hardware.reticle_compute_power
            A_ub_ = np.zeros(num_variables)
            b_ub_ = np.ones(1)
            for var, data_amount in ndata['compute_mark'].items():
                A_ub_[var] = data_amount / reticle_compute_power
            A_ub.append(A_ub_)
            b_ub.append(b_ub_)

        def add_transmission_constraint(edata):
            inter_reticle_bandwidth = self.hardware.inter_reticle_bandwidth
            A_ub_ = np.zeros(num_variables)
            b_ub_ = np.ones(1)
            for var, data_amount in edata['transmission_mark'].items():
                A_ub_[var] = data_amount / inter_reticle_bandwidth
            A_ub.append(A_ub_)
            b_ub.append(b_ub_)

        def add_dram_access_constraint(ndata):
            dram_bandwidth = self.hardware.dram_bandwidth
            A_ub_ = np.zeros(num_variables)
            b_ub_ = np.ones(1)
            for var, data_amount in ndata['dram_access_mark'].items():
                A_ub_[var] = data_amount / dram_bandwidth
            A_ub.append(A_ub_)
            b_ub.append(b_ub_)

        # add optimization constraint: f <= f_i
        for var in range(num_variables - 1):
            A_ub_ = np.zeros(num_variables)
            A_ub_[global_freq_index] = 1
            A_ub_[var] = -1
            b_ub_ = np.zeros(1)
            A_ub.append(A_ub_)
            b_ub.append(b_ub_)

        for node, ndata in G.nodes(data=True):
            add_compute_constraint(ndata)
            add_dram_access_constraint(ndata)

        for unode, vnode, edata in G.edges(data=True):
            add_transmission_constraint(edata)

        stack_func = lambda x: np.stack(x) if x else None

        linprog_kwargs = {
            'c': c,
            'A_ub': stack_func(A_ub),
            'b_ub': stack_func(b_ub),
            'A_eq': stack_func(A_eq),
            'b_eq': stack_func(b_eq),
            'bounds': bounds,
        }

        linprog_result = linprog(**linprog_kwargs)
        min_freq = linprog_result.x[global_freq_index]

        return min_freq
    
    def profile_utilization(self, group=True, per_module=False, per_task=False):
        logger.debug("Profiling resource utilization for lp solver")
        
        G = self.__build_annotated_graph()
        min_freq = self.__lp_solver(G)  # times / second

        group_compute_utils = []
        group_dram_bandwidth_utils = []
        group_inter_reticle_bandwidth_utils = []

        for node, ndata in G.nodes(data=True):
            if ndata['compute_mark']:
                reticle_compute_power = self.hardware.reticle_compute_power
                total_data_amount = sum(ndata['compute_mark'].values())
                total_util = total_data_amount*min_freq/reticle_compute_power
                group_compute_utils.append(total_util)
                if per_module:
                    logger.debug(f"Reticle coordinate {node}: compute_util={total_util:.2%}")
                if per_task:
                    for vrid, data_amount in ndata['compute_mark'].items():
                        logger.debug(f"- Reticle coordinate {node}: vrid={vrid}, compute_util={data_amount*min_freq/reticle_compute_power:.2%}")
            else:
                if ndata['reticle']: group_compute_utils.append(0)
            if ndata['dram_access_mark']:
                dram_bandwidth = self.hardware.dram_bandwidth
                total_data_amount = sum(ndata['dram_access_mark'].values())
                total_util = total_data_amount*min_freq/dram_bandwidth
                group_dram_bandwidth_utils.append(total_util)
                if per_module:
                    logger.debug(f"Reticle coordinate {node}: dram_bandwidth_util={total_util:.2%}")
                if per_task:
                    for vrid, data_amount in ndata['dram_access_mark'].items():
                        logger.debug(f"- Reticle coordinate {node}: vrid={vrid}, dram_bandwidth_util={data_amount*min_freq/dram_bandwidth:.2%}")
            else:
                if ndata['dram_port']: group_dram_bandwidth_utils.append(0)

        for u, v, edata in G.edges(data=True):
            if edata['transmission_mark']:
                inter_reticle_bandwidth = self.hardware.inter_reticle_bandwidth
                total_data_amount = sum(edata['transmission_mark'].values())
                total_util = total_data_amount*min_freq/inter_reticle_bandwidth
                group_inter_reticle_bandwidth_utils.append(total_util)
                if per_module:
                    logger.debug(f"Reticle link {u, v}: link_bandwidth={total_util:.2%}")
                if per_task:
                    for vrid, data_amount in edata['transmission_mark'].items():
                        logger.debug(f"- Reticle link {u, v}: vrid={vrid}, link_bandwidth_util={data_amount*min_freq/inter_reticle_bandwidth:.2%}")
            else:
                group_inter_reticle_bandwidth_utils.append(0)

        if group:
            logger.debug(f"Average compute util: {np.mean(group_compute_utils).item():.2%}")
            logger.debug(f"Maximum compute util: {np.max(group_compute_utils).item():.2%}")
            logger.debug(f"Average inter reticle bandwidth util: {np.mean(group_inter_reticle_bandwidth_utils).item():.2%}")
            logger.debug(f"Maximum inter reticle bandwidth util: {np.max(group_inter_reticle_bandwidth_utils).item():.2%}")
            logger.debug(f"Average DRAM bandwidth util: {np.mean(group_dram_bandwidth_utils).item():.2%}")
            logger.debug(f"Maximum DRAM bandwidth util: {np.max(group_dram_bandwidth_utils).item():.2%}")

        final_report = {
            'compute': np.max(group_compute_utils).item(),
            'inter_reticle': np.max(group_inter_reticle_bandwidth_utils).item(),
            'dram': np.max(group_dram_bandwidth_utils).item(),
        }
        return final_report

    def dump_graph(self):
        """ Dump graph and feature tensors for training GNN
        """
        G = self.__build_annotated_graph()
        min_freq = self.__lp_solver(G)

        # remove idle modules
        idle_links = [(u, v) for u, v, edata in G.edges(data=True) \
                       if not edata['transmission_mark']]
        G.remove_edges_from(idle_links)
        idle_nodes = [u for u, ndata in G.nodes(data=True) \
                      if not (ndata['compute_mark'] or ndata['dram_access_mark'])]
        G.remove_nodes_from(idle_nodes)

        # build a new heterogeneous graph from original graph
        # The effective bandwidth of a physical link is affected by:
        # - data transmission on this very link, and upstream/downstream transmission
        # - corresponding computation and dram access
        # Each physical link may have different effective bandwidth during simulation
        # so we want to formulate this problem as a node/edge regression task.

        # hyper nodes need to gather all relevant modules status
        num_hyper_nodes = len(self.vrid_2_var)

        # physical id -> heterogenenous graph index
        prids = [prid for prid, ndata in G.nodes(data=True) if ndata['compute_mark']]
        prid_2_hrid = {prid: i for i, prid in enumerate(prids)}
        pdpids = [pdpid for pdpid, ndata in G.nodes(data=True) if ndata['dram_access_mark']]
        pdpid_2_hdpid = {pdpid: i for i, pdpid in enumerate(pdpids)}
        plid_2_hlid = {plid: i for i, plid in enumerate(G.edges())}

        # reticle used by reticle task (hyper node)
        # feature is ideal latency (it is at least normalized...)
        hrid_used_by_hyper = []
        hrid_used_by_hyper_features = []
        compute_power = self.hardware.reticle_compute_power
        for prid, hrid in prid_2_hrid.items():
            compute_mark = G.nodes[prid]['compute_mark']
            for hyper, data_amount in compute_mark.items():
                hrid_used_by_hyper.append((hrid, hyper))
                hrid_used_by_hyper_features.append(data_amount / compute_power)

        hdpid_used_by_hyper = []
        hdpid_used_by_hyper_features = []
        dram_bandwidth = self.hardware.dram_bandwidth
        for pdpid, hdpid in pdpid_2_hdpid.items():
            dram_access_mark = G.nodes[pdpid]['dram_access_mark']
            for hyper, data_amount in dram_access_mark.items():
                hdpid_used_by_hyper.append((hdpid, hyper))
                hdpid_used_by_hyper_features.append(data_amount / dram_bandwidth)
        
        hlid_used_by_hyper = []
        hlid_used_by_hyper_features = []
        hlid_label = []  # effective bandwidth to predict, 0 ~ 100%
        inter_reticle_bandwidth = self.hardware.inter_reticle_bandwidth
        for plid, hlid in plid_2_hlid.items():
            transmission_mark = G.edges[plid]['transmission_mark']
            for hyper, data_amount in transmission_mark.items():
                hlid_used_by_hyper.append((hlid, hyper))
                hlid_used_by_hyper_features.append(data_amount / inter_reticle_bandwidth)
            total_data_amount = sum(transmission_mark.values())
            effective_bandwidth = total_data_amount * min_freq / inter_reticle_bandwidth
            hlid_label.append(effective_bandwidth)

        # add edges between links
        # is u -> v -> r, then we say (u, v) goes to (v, r)
        hlid_goes_to_hlid = []
        hlid_connect_to_ratio = []  # how much data of prev link are transferred to next link
        hlid_connected_by_ratio = []  # how much data of succ link are transferred from prev link
        # this is not very accurate, since we've already mixed up all the transmission of the same fused task
        # therefore, we consider the maximum possible transmission inside this router
        for (u, v), (r, s) in product(G.edges(), G.edges()):
            if v == r:
                hlid_src = plid_2_hlid[(u, v)]
                hlid_dst = plid_2_hlid[(r, s)]
                hlid_goes_to_hlid.append((hlid_src, hlid_dst))
                total_data_amount = 0
                prev_transmission_mark = G.edges[(u, v)]['transmission_mark']
                succ_transmission_mark = G.edges[(r, s)]['transmission_mark']
                union_hyper_set = set(prev_transmission_mark.keys()) | set(succ_transmission_mark.keys())
                for hyper in union_hyper_set:
                    data_amount = min(prev_transmission_mark.get(hyper, 0), succ_transmission_mark.get(hyper, 0))
                    total_data_amount += data_amount
                prev_total_data_amount = sum(prev_transmission_mark.values())
                succ_total_data_amount = sum(succ_transmission_mark.values())
                hlid_connect_to_ratio.append(total_data_amount / prev_total_data_amount)
                hlid_connected_by_ratio.append(total_data_amount / succ_total_data_amount)

        def decompose_edge_list(edge_list, reverse=False):
            src_list = th.tensor([e[0] for e in edge_list])
            dst_list = th.tensor([e[1] for e in edge_list])
            if not reverse:
                return src_list, dst_list
            else:
                return dst_list, src_list

        # build heterogeneous graph
        data_dict = {
            ('reticle', 'reticle_used_by', 'task'): decompose_edge_list(hrid_used_by_hyper),
            ('dram_port', 'dram_port_used_by', 'task'): decompose_edge_list(hdpid_used_by_hyper),
            ('link', 'link_used_by', 'task'): decompose_edge_list(hlid_used_by_hyper),

            ('task', 'use_reticle', 'reticle'): decompose_edge_list(hrid_used_by_hyper, reverse=True),
            ('task', 'use_dram_port', 'dram_port'): decompose_edge_list(hdpid_used_by_hyper, reverse=True),
            ('task', 'use_link', 'link'): decompose_edge_list(hlid_used_by_hyper, reverse=True),

            ('link', 'connect_to', 'link'): decompose_edge_list(hlid_goes_to_hlid),
            ('link', 'connected_by', 'link'): decompose_edge_list(hlid_goes_to_hlid, reverse=True),
        }
        def generate_feature_tensor(feat, add_ones=True):
            t = th.tensor(feat).reshape(-1, 1)
            if add_ones:
                ones = th.ones(len(feat), 1)
                t = th.concat((t, ones), dim=-1)
            return t

        feat_dict = {
            'reticle_used_by': generate_feature_tensor(hrid_used_by_hyper_features),
            'dram_port_used_by': generate_feature_tensor(hdpid_used_by_hyper_features),
            'link_used_by': generate_feature_tensor(hlid_used_by_hyper_features),

            'use_reticle': generate_feature_tensor(hrid_used_by_hyper_features),
            'use_dram_port': generate_feature_tensor(hdpid_used_by_hyper_features),
            'use_link': generate_feature_tensor(hlid_used_by_hyper_features),

            'connect_to': generate_feature_tensor(hlid_connect_to_ratio, add_ones=False),
            'connected_by': generate_feature_tensor(hlid_connected_by_ratio, add_ones=False),
        }
        label_dict = {
            'link': th.tensor(hlid_label),
        }

        assert hlid_label, "We don't support no-link cases"

        return data_dict, feat_dict, label_dict