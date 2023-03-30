
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from copy import deepcopy
import networkx as nx
from networkx import DiGraph
from scipy.optimize import linprog
import numpy as np

from dse4wse.pe_graph.hardware import WaferScaleEngine
from dse4wse.pe_graph.task import BaseWaferTask, ListWaferTask, ComputeReticleTask, DramAccessReticleTask, PeerAccessReticleTask, FusedReticleTask
from dse4wse.pe_graph.mapper import WseMapper

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

        for reticle_task in self.task:
            assert reticle_task.task_type == 'fused'
            vrid = reticle_task.virtual_reticle_id
            physical_reticle_coordinate = self.mapper.find_physical_reticle_coordinate(vrid)

            for subtask in reticle_task.get_subtask_list():
                if subtask.task_type == 'compute':
                    subtask: ComputeReticleTask
                    ndata = G.nodes[physical_reticle_coordinate]
                    ndata['compute_mark'][vrid] = subtask.compute_amount

                elif subtask.task_type == 'dram_access':
                    subtask: DramAccessReticleTask
                    physical_dram_port_coordinate = self.mapper.find_physical_dram_port_coordinate(subtask.virtual_dram_port)
                    # mark dram port access amount
                    ndata = G.nodes[physical_dram_port_coordinate]
                    ndata['dram_access_mark'][vrid] = subtask.data_amount
                    # mark every link's transmission
                    if subtask.access_type == 'read':
                        link_list = self.mapper.find_read_dram_routing_path(physical_reticle_coordinate, physical_dram_port_coordinate)
                    else:
                        link_list = self.mapper.find_write_dram_routing_path(physical_reticle_coordinate, physical_dram_port_coordinate)
                    for link in link_list:
                        edata = G.edges[link]
                        edata['transmission_mark'][vrid] = subtask.data_amount

                else:
                    raise NotImplementedError(f"Unrecognized subtask type {subtask.type}")
        return G
        
    def __lp_solver(self, G: DiGraph) -> float:
        """
        Calculate the slowest frequency of all reticle tasks            
        """
        global_freq_index = len(self.task)
        num_variables = len(self.task) + 1

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
            b_ub_ = np.ones(1) * reticle_compute_power
            for vrid, data_amount in ndata['compute_mark'].items():
                A_ub_[vrid] = data_amount
            A_ub.append(A_ub_)
            b_ub.append(b_ub_)

        def add_transmission_constraint(edata):
            inter_reticle_bandwidth = self.hardware.inter_reticle_bandwidth
            A_ub_ = np.zeros(num_variables)
            b_ub_ = np.ones(1) * inter_reticle_bandwidth
            for vrid, data_amount in edata['transmission_mark'].items():
                A_ub_[vrid] = data_amount
            A_ub.append(A_ub_)
            b_ub.append(b_ub_)

        def add_dram_access_constraint(ndata):
            dram_bandwidth = self.hardware.dram_bandwidth
            A_ub_ = np.zeros(num_variables)
            b_ub_ = np.ones(1) * dram_bandwidth
            for vrid, data_amount in ndata['dram_access_mark'].items():
                A_ub_[vrid] = data_amount
            A_ub.append(A_ub_)
            b_ub.append(b_ub_)

        # add optimization constraint: f <= f_i
        for var in range(num_variables - 1):
            A_ub_ = np.zeros(num_variables)
            A_ub_[global_freq_index] = 1
            A_ub_[var] = -1
            b_ub_ = np.ones(1)
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
        logger.debug(linprog_kwargs)

        linprog_result = linprog(**linprog_kwargs)
        min_freq = linprog_result.x[global_freq_index]
        return min_freq