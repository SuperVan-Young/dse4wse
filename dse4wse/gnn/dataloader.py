
import dgl
from dgl.data import DGLDataset
import os
import pickle as pkl
import torch

class LinkUtilDataset(DGLDataset):
    def __init__(self, url=None, raw_dir=None, save_dir=None, hash_key=..., force_reload=False, verbose=False, transform=None):
        super().__init__(name='link_util',
                         url=url, 
                         raw_dir=raw_dir, 
                         save_dir=save_dir, 
                         hash_key=hash_key, 
                         force_reload=force_reload, 
                         verbose=verbose, 
                         transform=transform)
        self.data_names = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]

    def __getitem__(self, idx):
        data_path = os.path.join(self.save_dir, self.data_names[idx])
        with open(data_path, 'rb') as f:
            graph_data, feat_dict, label_dict = pkl.load(f)
        G = dgl.heterograph(graph_data)
        for edge_type, edge_data in feat_dict.items():
            G.edges[edge_type].data['feat'] = edge_data
        for node_type, node_label in label_dict.items():
            G.nodes[node_type].data['label'] = node_label
        return G

    def __len__(self):
        num_files = len(self.data_names)
        return num_files
    
class NoCeptionDataset(DGLDataset):
    def __init__(self, url=None, raw_dir=None, save_dir=None, hash_key=..., force_reload=False, verbose=False, transform=None):
        name = 'NoCeption Dataset'
        super().__init__(name, url, raw_dir, save_dir, hash_key, force_reload, verbose, transform)
        self.data_names = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]

    def __getitem__(self, idx):
        data_path = os.path.join(self.save_dir, self.data_names[idx])
        with open(data_path, 'rb') as f:
            edge_srcs, edge_dsts, node_feats, edge_feats, end2end_latency = pkl.load(f)
        graph_data = (edge_srcs, edge_dsts)
        G = dgl.graph(graph_data)
        G.ndata['inp'] = torch.tensor(node_feats, dtype=torch.float32)
        G.edata['inp'] = torch.tensor(edge_feats, dtype=torch.float32)
        label = torch.tensor(end2end_latency)
        return G, label
    
    def __len__(self):
        num_files = len(self.data_names)
        return num_files