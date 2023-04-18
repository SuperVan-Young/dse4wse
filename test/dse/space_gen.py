import json
import math
import os
from itertools import product
from typing import List
# import pandas as pd
from tqdm import tqdm

DSE_DIR = os.path.dirname(os.path.abspath(__file__))

class design_space_construction():
    def __init__(self):
        self.design_space = json.load(open(os.path.join(DSE_DIR, 'design_space.json'), "r"))
        self.sram_table = json.load(open(os.path.join(DSE_DIR, 'sram_table.json'), "r"))
        self.noc_table = json.load(open(os.path.join(DSE_DIR, 'noc_table.json'), "r"))

        # Wikichips
        self.reticle_limit_height = 33000
        self.reticle_limit_width = 26000
        self.wafer_limit = 215000

        self.core_gap = 60
        self.reticle_gap = 800

        self.dojo_overhead = 0

    def gen_parameter_list(self):
        # print("Genetating Parameter List")
        self.parameter_list = {}
        for key, value in self.design_space.items():
            self.parameter_list[key] = {}
            for component_key, component_value in self.design_space[key].items():
                # print(component_key, component_value)
                self.parameter_list[key][component_key] = []
                init = component_value['initial value']
                end = component_value['end value']
                style = component_value['style']
                scale = component_value['scale']
                res = init
                while res <= end:
                    self.parameter_list[key][component_key].append(res)
                    if style == 'linear':
                        res += scale
                    elif style == 'exponential':
                        res *= scale
        print(self.parameter_list)

    def gen_design_points(self):
        total_design_points = 0
        f = open("design_points.list", "w")
        # df = pd.DataFrame(columns=[
        #     "core_buffer_size", 
        #     "core_buffer_bw", 
        #     "core_mac_num", 
        #     "core_noc_bw", 
        #     "core_noc_vc", 
        #     "core_noc_buffer_size", 
        #     "reticle_bw", 
        #     "core_array_h", 
        #     "core_array_w", 
        #     "reticle_array_h", 
        #     "reticle_array_w",
        #     "wafer_mem_bw",
        # ])

        for core_buffer_size, core_mac_num in product(self.parameter_list['Core']['buffer size'] + [48], self.parameter_list['Core']['MAC number']):
            core_buffer_size = int(core_buffer_size)
            if str(core_buffer_size) in self.sram_table:
                core_buffer_bw_list = [int(core_buffer_bw) for core_buffer_bw in self.sram_table[str(core_buffer_size)] \
                                       if int(core_buffer_bw) in self.parameter_list['Core']['buffer bandwidth']]
            else: continue   
            for core_buffer_bw, \
                core_noc_bw, \
                core_noc_vc, \
                core_noc_buffer_size, \
                reticle_bw, \
                wafer_mem_bw, \
                in product(
                    core_buffer_bw_list, \
                    self.parameter_list['Core']['NoC bandwidth'], \
                    self.parameter_list['Core']['NoC router virtual channel'], \
                    self.parameter_list['Core']['NoC router buffer size'], \
                    self.parameter_list['Reticle']['inter-reticle bandwidth'], \
                    self.parameter_list['Wafer']['off-chip memory bandwidth'], \
                ):
                    sram_area = self.get_sram_area(core_buffer_size, core_buffer_bw)
                    logic_area = self.get_logic_area(core_mac_num)
                    noc_area = self.get_noc_area(core_noc_bw, core_noc_vc, core_noc_buffer_size)
                    noc_h = noc_w = math.sqrt(noc_area)

                    core_area = sram_area[2] + logic_area
                    
                    core_h = max(sram_area[0], sram_area[1])
                    core_w = core_area / core_h
                    core_h += noc_h
                    core_w += noc_w 

                    # debug, really big step
                    for core_array_h, core_array_w in product(range(8, 129, 4), range(8, 129, 4)):
                        reticle_h = core_array_h * (core_h + self.core_gap)
                        reticle_w = core_array_w * (core_w + self.core_gap)
                        if reticle_h < (self.reticle_limit_height - self.dojo_overhead) and reticle_w < (self.reticle_limit_width - self.dojo_overhead):
                            max_reticle_array_h = int(self.wafer_limit / (reticle_h + self.reticle_gap))
                            max_reticle_array_w = int(self.wafer_limit / (reticle_w + self.reticle_gap))

                            for reticle_array_h, reticle_array_w in product(range(1, max_reticle_array_h + 1, 1), range(1, max_reticle_array_w + 1, 1)):
                                design_point = [
                                    core_buffer_size, 
                                    core_buffer_bw, 
                                    core_mac_num, 
                                    core_noc_bw, 
                                    core_noc_vc, 
                                    core_noc_buffer_size, 
                                    reticle_bw, 
                                    core_array_h, 
                                    core_array_w,
                                    wafer_mem_bw, 
                                    reticle_array_h, 
                                    reticle_array_w
                                ]
                                print(design_point, file=f)
                                total_design_points += 1
                                print(total_design_points)

        #                     design_point = {
        #                         "core_buffer_size": core_buffer_size, 
        #                         "core_buffer_bw": core_buffer_bw, 
        #                         "core_mac_num": core_mac_num, 
        #                         "core_noc_bw": core_noc_bw, 
        #                         "core_noc_vc": core_noc_vc, 
        #                         "core_noc_buffer_size": core_noc_buffer_size, 
        #                         "reticle_bw": reticle_bw, 
        #                         "core_array_h": core_array_h, 
        #                         "core_array_w": core_array_w, 
        #                         "reticle_array_h": reticle_array_h, 
        #                         "reticle_array_w": reticle_array_w,
        #                         "wafer_mem_bw": wafer_mem_bw,
        #                     }
        #                     df.loc[len(df.index)] = design_point
        #                     total_design_points += 1
        #                     if total_design_points % 1000 == 0:
        #                         print(design_point)
        #                         print(f"Total design point reaches {total_design_points}")
        #                         df.to_excel("design_points.xlsx", index=False)

        # print(f"#Design_Points = {len(df.index)}")
        print("Total design point number: ", total_design_points)

        # df.to_excel("design_points.xlsx", index=False)

    def get_sram_area(self, core_buffer_size, core_buffer_bw):
        sram_compiler_result = self.sram_table[str(core_buffer_size)][str(core_buffer_bw)]
        return [sram_compiler_result["height"] * 0.63, sram_compiler_result["width"] * 0.63, sram_compiler_result["area"] * (0.63 ** 2)]

    def get_logic_area(self, core_mac_num):
        return core_mac_num * 4360 * 2 * (0.63 ** 3)
        # return 1.0

    def get_noc_area(self, core_noc_bw, core_noc_vc, core_noc_buffer_size):
        return self.noc_table[str(core_noc_bw)][str(core_noc_vc)][str(core_noc_buffer_size)]["area"] * (0.63 ** 3) 
        # return 1.0

    def run(self):
        print("Generating Valid Design Points")
        self.gen_parameter_list()
        self.gen_design_points()

def main():
    DSC = design_space_construction()
    DSC.run()

if __name__ == '__main__':
    main()