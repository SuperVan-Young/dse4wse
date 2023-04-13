import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dse4wse.gnn.dataloader import LinkUtilDataset
from dse4wse.gnn.model import HeteroNet

def run_model():
    dataset = LinkUtilDataset(save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    test_data = dataset[0]

    model = HeteroNet(h_dim=64)
    model(test_data)

if __name__ == "__main__":
    run_model()