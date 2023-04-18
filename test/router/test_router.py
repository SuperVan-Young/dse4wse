import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dse4wse.pe_graph.mapper.reticle_router import XYReticleRouter
from dse4wse.utils import logger

def test_router(src, dst):
    logger.info(f"src = {src}, dst = {dst}")
    router = XYReticleRouter()
    link_list = router(src, dst)
    logger.info(link_list)
    
TEST_CASES = [
    ((0, 0), (0, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 2)),
    ((0, 0), (0, -1)),
    ((0, 0), (0, -2)),
    ((0, 0), (1, 0)),
    ((0, 0), (2, 0)),
    ((0, 0), (-1, 0)),
    ((0, 0), (-2, 0)),
    ((1, 2), (3, 4)),
]

def run_all_tests():
    for case in TEST_CASES:
        test_router(*case)

if __name__ == "__main__":
    run_all_tests()