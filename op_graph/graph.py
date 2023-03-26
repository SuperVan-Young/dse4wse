import os
import sys
from typing import Dict, Union, List
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np
from networkx import DiGraph
from itertools import chain

from op import Operator
from utils import logger, ArchConfig, TensorInfo, calc_comm_cost_on_same_devices, calc_comm_cost_on_disjoint_devices

class OpGraph(DiGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.name = None
        self._duplication_table = None
        self._onnx_graph = None

    @property
    def duplication_table(self):
        if self._duplication_table is None:
            return {name: 1 for name in self.nodes()}
        else:
            return self._duplication_table
        
    def get_tensors(self, kind=['weight', 'input', 'output', 'activation', 'constant']) -> Dict[str, TensorInfo]:
        tensors = {}
        for op_name, op in self.nodes(data='operator'):
            op: Operator
            for tensor_name, tensor in chain(op.input_tensors.items(), op.output_tensors.items()):
                if tensor.kind in kind:
                    tensors[tensor_name] = tensor
        return tensors
    
    def get_propagation_latency(self, arch_config: ArchConfig, forward=True) -> float:
        """Sequentially compute each operation and transmit data between operators
        """
        total_compute_latency = 0
        total_boxing_latency = 0

        for op_name, op in self.nodes(data='operator'):
            op: Operator
            if forward:
                compute_latency = op.get_fp_latency(op.final_intra_sbp_sigs, arch_config)
            else:
                compute_latency = op.get_bp_latency(op.final_intra_sbp_sigs, arch_config)
            total_compute_latency += compute_latency
        for prev, succ in self.edges():
            total_boxing_latency += self.get_boxing_latency(prev, succ, arch_config)

        total_latency = total_compute_latency + total_boxing_latency
        return total_latency

    def get_boxing_latency(self, prev: str, succ: str, arch_config: ArchConfig) -> float:
        edata = self.edges[prev, succ]
        common_tensor_names = edata['common_tensor_names']
        boxing_prev_index = edata['boxing_prev_index']
        boxing_succ_index = edata['boxing_succ_index']

        prev_op = self.nodes[prev]['operator']
        succ_op = self.nodes[succ]['operator']

        for common_tensor_name in common_tensor_names:
            prev_local_name = boxing_prev_index[common_tensor_name]
            succ_local_name = boxing_succ_index[common_tensor_name]
            prev_inter_sbp_sig = prev_op.final_inter_sbp_sigs[prev_local_name]
            succ_inter_sbp_sig = succ_op.final_inter_sbp_sigs[succ_local_name]
            tensor = prev_op.output_tensors[prev_local_name]

            if prev_inter_sbp_sig.placement == succ_inter_sbp_sig.placememt:
                return calc_comm_cost_on_same_devices(tensor, prev_inter_sbp_sig, succ_inter_sbp_sig, arch_config)
            else:
                raise NotImplementedError
                return calc_comm_cost_on_disjoint_devices()
            
    def get_inter_layer_sbp_signatures(self, node):
        """ Get input tensor's sbp signature on previous tensors
        """
        # FIXME: use local name! unify representation with BaseOperator
        inter_layer_sbp_signatures = {}

        cur_op = self.nodes[node]['operator']
        cur_op: Operator
        input_tensor_global_names = [tensor_info.name for tensor_info in cur_op.input_tensors.values()]
        for prev in self.predecessors(node):
            prev_op = self.nodes[prev]['operator']
            prev_op: Operator
            for local_name, tensor_info in prev_op.output_tensors.items():
                global_name = tensor_info.name
                if global_name in input_tensor_global_names:
                    inter_layer_sbp_signatures[global_name] = prev_op.final_sbp_signatures[local_name]
        return inter_layer_sbp_signatures

    def profile_core_allocation(self):
        logger.info(f"Profiling core allocation of OP graph {self.name}")
        for node, op in self.nodes(data='operator'):
            op: Operator
            logger.info(f"{node}: {op.num_core_range}")
            
            upper_bound = max([x.sup for x in op.num_core_range.extrema])
            if upper_bound == 0:
                logger.warn(f"{node} doesn't have allocated cores.")

    def profile_final_sbp_signatures(self):
        logger.info(f"Profiling final SBP signatures of OP graph {self.name}")
        for node, op in self.nodes(data='operator'):
            op: Operator
            logger.info(f"{node}: {op.final_sbp_signatures}")

    def profile_performance(self, arch_config: ArchConfig, in_detail=False):
        logger.info(f"Profiling performance of OP graph {self.name}")

        duplication_table = self.duplication_table

        throughput_bottleneck_name = None
        throughput_bottleneck_latency = 0

        create_empty_report = lambda: {
            'comm_input_cost': 0,
            'comm_reduce_cost': 0,
            'compute_cost': 0,
        }
        overall_report_summary = create_empty_report()
        op_category_2_report_summary = {}

        for node, op in self.nodes(data='operator'):
            op: Operator

            inter_layer_sbp_signatures = self.get_inter_layer_sbp_signatures(node)
            latency_report = op.estimate_cost(op.final_sbp_signatures, arch_config, inter_layer_sbp_signatures, detailed_report=True)

            if in_detail:
                logger.info(f"Operator: {node}")
                for name, tensor_info in op.input_tensors.items():
                    logger.info(f"    [Input] {name}: {tensor_info.name}")
                for name, tensor_info in op.input_tensors.items():
                    logger.info(f"    [Output] {name}: {tensor_info.name}")
                for name, tensor_info in chain(op.input_tensors.items(), op.output_tensors.items()):
                    logger.info(f"    [Shape] {name}: {tensor_info.shape}")
                for name, sbp in op.final_sbp_signatures.items():
                    logger.info(f"    [SBP] {name}: {sbp}")
                logger.info(f"    [Cost] comm input:  {int(latency_report['comm_input_cost']):>10}")
                logger.info(f"    [Cost] comm reduce: {int(latency_report['comm_reduce_cost']):>10}")
                logger.info(f"    [Cost] compute:     {int(latency_report['compute_cost']):>10}")

            # TODO: intra-layer pipelining
            total_latency = int(latency_report['comm_input_cost'] + latency_report['comm_reduce_cost'] + latency_report['compute_cost'])
            if total_latency > throughput_bottleneck_latency:
                throughput_bottleneck_name = op.name
                throughput_bottleneck_latency = total_latency

            dup = duplication_table[node]
            op_category = type(op)
            if op_category not in op_category_2_report_summary:
                op_category_2_report_summary[op_category] = create_empty_report()
            report_summary = op_category_2_report_summary[op_category]
            for k in report_summary:
                report_summary[k] += latency_report[k] * dup
            for k in overall_report_summary:
                overall_report_summary[k] += latency_report[k] * dup

        logger.info("#" * 20 + "   Summary   " + "#" * 20)
        logger.info(f"Throughput bottleneck: {throughput_bottleneck_name}, {int(throughput_bottleneck_latency)} cycles")

        overall_total_latency = sum(overall_report_summary.values())
        logger.info(f"Overall latency breakdown by cost type:")
        for k, v in overall_report_summary.items():
            logger.info(f"    {k:<20} {int(v):>10} cycles ({v / overall_total_latency:.2%})")
        logger.info(f"    {'Total':<20} {int(overall_total_latency):>10} cycles")

        for category, report in op_category_2_report_summary.items():
            category_total_latency = sum(report.values())
            logger.info(f"{category} latency breakdown by cost type:")
            for k, v in report.items():
                logger.info(f"    {k:<20} {int(v):>10} cycles ({v / category_total_latency:.2%})")
            logger.info(f"    {'Total':<20} {int(category_total_latency):>10} cycles")
        
        logger.info("Overall latency breakdown by operator category:")
        for cost_type, cost_value in overall_report_summary.items():
            logger.info(f"Cost type: {cost_type}")
            for category, report in op_category_2_report_summary.items():
                logger.info(f"    {category}: {int(report[cost_type]):>10} cycles ({report[cost_type] / cost_value:>.2%})")
            logger.info(f"    Total: {int(cost_value):>10} cycles")

        assert total_latency < np.inf

        return total_latency
    
def build_op_graph_from_operator_list(operators: List[Operator]):
    op_graph = OpGraph()

    for op in operators:
        name = op.name
        op_graph.add_node(name)
        op_graph.nodes[name]['operator'] = op

    # connect edges
    for u, u_op in op_graph.nodes(data='operator'):
        u_op: Operator
        u_out_tensor_name_2_local_name = {tensor.name: local_name for local_name, tensor in u_op.output_tensors.items()}
        for v, v_op in op_graph.nodes(data='operator'):
            v_op: Operator
            v_in_tensor_name_2_local_name = {tensor.name: local_name for local_name, tensor in v_op.input_tensors.items()}
            common_tensor_names = set(u_out_tensor_name_2_local_name.keys()) & set(v_in_tensor_name_2_local_name.keys())
            if common_tensor_names:
                # check if common tensors have the same shape
                for common_tensor_name in common_tensor_names:
                    perv_tensor = u_op.output_tensors[u_out_tensor_name_2_local_name[common_tensor_name]]
                    succ_tensor = v_op.input_tensors[v_in_tensor_name_2_local_name[common_tensor_name]]
                    assert tuple(perv_tensor.shape) == tuple(succ_tensor.shape)
                boxing_prev_index = {tensor_name: local_name for tensor_name, local_name in u_out_tensor_name_2_local_name.items() if tensor_name in common_tensor_names}
                boxing_succ_index = {tensor_name: local_name for tensor_name, local_name in v_in_tensor_name_2_local_name.items() if tensor_name in common_tensor_names}
                op_graph.add_edge(u, v, boxing_prev_index=boxing_prev_index, boxing_succ_index=boxing_succ_index, common_tensor_names=common_tensor_names)

    return op_graph