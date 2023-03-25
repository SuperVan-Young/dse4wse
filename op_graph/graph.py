import os
import sys
from typing import Dict, Union
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np
from networkx import DiGraph
from itertools import chain

from op import Operator
from utils import logger, ArchConfig

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