
import os
import json
from typing import Tuple
from functools import reduce
import operator
import math

MAC_DYNAMIC_ENERGY = (35.3 + 16.679) * (0.75 ** 2)      # pJ
MAC_STATIC_POWER = (0.12 + 0.053) * 1e-3 * (0.75 ** 2)  # W

NOC_ROUTER_STATIC_POWER_TABLE = {
    32: 0.005060908125000001,
    64: 0.006567693750000001,
    128: 0.0095812875,
    256: 0.015608418749999999,
    512: 0.02766268125,
    1024: 0.0517712625,
    2048: 0.09998831250000001,
    4096: 0.19642274999999998,
}  # W

NOC_CHANNEL_FACTOR = 0.15                # pJ / bit / mm

CEREBRAS_RETICLE_CHANNEL_ENERGY = 0.25   # pJ / bit
DOJO_RETICLE_CHANNEL_ENERGY = 1.25       # pJ / bit

DRAM_ACCESS_ENERGY = 8.75  # pJ / bit

SRAM_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sram_table.json')
NOC_TABLE_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noc_table.json')

class WsePowerTable():

    def __init__(self,
                 core_buffer_size: int,
                 core_buffer_bw: int,
                 core_mac_num: int,
                 core_noc_bw: int,
                 core_noc_vc: int,
                 core_noc_buffer_size: int,
                 reticle_bw: float,
                 core_array_h: int,
                 core_array_w: int,
                 wafer_mem_bw: int,
                 reticle_array_h: int,
                 reticle_array_w: int,
                 package_type: str = 'cerebras',
                 ) -> None:
        self.core_buffer_size = core_buffer_size
        self.core_buffer_bw = core_buffer_bw
        self.core_mac_num = core_mac_num
        self.core_noc_bw = core_noc_bw
        self.core_noc_vc = core_noc_vc
        self.core_noc_buffer_size = core_noc_buffer_size
        self.reticle_bw = reticle_bw
        self.core_array_h = core_array_h
        self.core_array_w = core_array_w
        self.wafer_mem_bw = wafer_mem_bw
        self.reticle_array_h = reticle_array_h
        self.reticle_array_w = reticle_array_w
        self.package_type = package_type

        # load various table
        self.sram_table = json.load(open(SRAM_TABLE_PATH, 'r'))
        self.noc_table  = json.load(open(NOC_TABLE_PATH, 'r'))

    def get_compute_power(self, data_amount: int, total_time: int) -> float:
        """ Power consumption of computation.
        """
        static_power = self._get_static_compute_power()
        dynamic_power = self._get_dynamic_compute_power(data_amount, total_time)
        total_power = static_power + dynamic_power
        return total_power
    
    def get_interconnect_power(self, data_amount: int, total_time: int) -> float:
        """ Power consumption of inter-reticle interconnection
        Each inter-reticle link is composed of N routers and N inter-core link, and a inter-reticle channel
        N is expected to be the arithmetic average of reticle height & width.
        """
        # router power
        static_router_power = self._get_static_router_power()
        dynamic_router_power = self._get_dynamic_router_power(data_amount, total_time)
        total_power = static_router_power + dynamic_router_power
        return total_power
    
    def get_sram_access_power(self, data_amount: int, total_time: int) -> float:
        """ Power consumption of SRAM access.
        Data amount is a inferred from FLOPS and arithmetic intensity.
        """
        static_sram_power = self._get_static_sram_power()
        dynamic_sram_power = self._get_dynamic_sram_power(data_amount, total_time)
        total_power = static_sram_power + dynamic_sram_power
        return total_power
    
    def get_dram_access_power(self, data_amount: int, total_time: int) -> float:
        """ Power consumption of DRAM access.
        We only consider dynamic access energy.
        """
        total_power = DRAM_ACCESS_ENERGY * (data_amount * 8) / total_time * 1e12
        return total_power

    def _get_static_compute_power(self) -> float:
        num_mac = self.core_mac_num * self._get_num_total_cores()
        static_power = num_mac * MAC_STATIC_POWER
        return static_power
    
    def _get_dynamic_compute_power(self, data_amount: int, total_time: int) -> float:
        dynamic_power = data_amount * MAC_DYNAMIC_ENERGY / total_time * 1e12
        return dynamic_power
    
    def _get_static_router_power(self) -> float:
        num_total_router = self._get_num_total_cores()
        router_static_power = NOC_ROUTER_STATIC_POWER_TABLE[self.core_noc_bw]
        total_power = num_total_router * router_static_power
        return total_power
    
    def _get_dynamic_router_power(self, data_amount: int, total_time: int) -> float:
        core_h, core_w = self._get_core_height_and_width()
        core_gap = 60
        reticle_h = (core_h + core_gap) * self.core_array_h
        reticle_w = (core_w + core_gap) * self.core_array_w
        expected_link_length = (reticle_h + reticle_w) / 2
        intra_reticle_power = NOC_CHANNEL_FACTOR * expected_link_length * (data_amount * 8) / total_time * 1e12

        if self.package_type == 'cerebras':
            reticle_channel_energy = CEREBRAS_RETICLE_CHANNEL_ENERGY
        elif self.package_type == 'dojo':
            reticle_channel_energy = DOJO_RETICLE_CHANNEL_ENERGY
        else:
            raise NotImplementedError
        inter_reticle_power = reticle_channel_energy * (data_amount * 8) / total_time * 1e12

        total_power = intra_reticle_power + inter_reticle_power
        return total_power
    
    def _get_num_total_cores(self) -> int:
        return reduce(operator.mul, [
            self.core_array_h,
            self.core_array_w,
            self.reticle_array_h,
            self.reticle_array_w
        ], initial=1)

    def _get_core_height_and_width(self) -> Tuple[int, int]:
        """ Copied from space_gen, verified with zjc.
        """
        sram_compiler_result = self.sram_table[str(self.core_buffer_size)][str(self.core_buffer_bw)]
        sram_height = sram_compiler_result['height'] * 0.63
        sram_width = sram_compiler_result['width'] * 0.63
        sram_area = sram_compiler_result['area'] * (0.63 ** 2)

        logic_area = self.core_mac_num * 4360 * 2 * (0.63 ** 3)

        noc_area = self.noc_table[str(self.core_noc_bw)][str(self.core_noc_vc)][str(self.core_noc_buffer_size)]['area'] * (0.63 ** 3)
        noc_height = noc_width = math.sqrt(noc_area)

        core_area = sram_area + logic_area
        core_h = max(sram_height, sram_width)
        core_w = core_area / core_h
        core_h += noc_height
        core_w += noc_width

        return core_h, core_w
    
    def _get_static_sram_power(self) -> float:
        num_total_sram = self._get_num_total_cores()
        sram_static_power = self.sram_table[str(self.core_buffer_size)][str(self.core_buffer_bw)]['static_power'] * 1e-3
        total_power = num_total_sram * sram_static_power
        return total_power
    
    def _get_dynamic_sram_power(self, data_amount: int, total_time: int) -> float:
        sram_compiler_result = self.sram_table[str(self.core_buffer_size)][str(self.core_buffer_bw)]
        sram_access_energy = (sram_compiler_result['read_power'] + sram_compiler_result['write_power']) / 2
        total_power = sram_access_energy * (data_amount * 8) / total_time * 1e12
        return total_power
    