
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

from dse4wse.utils import logger

class TaskInfoConv(nn.Module):
    """ Gather task info from edges to modules, and aggregate them on task hyper node.
    """
    def __init__(self, 
                 h_dim: int = 64,
                 activate_inp: bool = False,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.h_dim = h_dim
        self.init_linears = {
            'reticle': nn.Linear(2, h_dim),  # append an auxillary 1 in the end
            'dram_port': nn.Linear(2, h_dim),
            'link': nn.Linear(2, h_dim),
        }
        self.task_linear = nn.Linear(3 * h_dim, h_dim)
        self.link_linear = nn.Linear(h_dim, h_dim)
        self.activation_func = nn.GELU()
        self.task_norm = nn.LayerNorm(h_dim)
        self.link_norm = nn.LayerNorm(h_dim)
        
    def _init_inp(self, G: dgl.heterograph):
        # sum workload of all task's workload on the module
        update_func_tuple = (fn.copy_e('feat', 'm'), fn.sum('m', 'inp'))
        G.multi_update_all(
            {
                'use_reticle': update_func_tuple,  
                'use_dram_port': update_func_tuple,
                'use_link': update_func_tuple,
            },
            'sum',
        )
        for ntype in ['reticle', 'dram_port', 'link']:
            linear = self.init_linears[ntype]
            G.nodes[ntype].data['h'] = torch.tanh(linear(G.nodes[ntype].data['inp']))

    def _task_update_h(self, G: dgl.heterograph):
        # task only cares about the hottest spot
        update_func_tuple = (fn.copy_u('h', 'm'), fn.max('m', 'h'))

        def cross_reducer(l):
            t = torch.concat(l, dim=-1)
            t = self.task_linear(t)
            t = self.activation_func(t)
            t = self.task_norm(t)
            return t

        G.multi_update_all(
            {
                'reticle_used_by': update_func_tuple,  
                'dram_port_used_by': update_func_tuple,
                'link_used_by': update_func_tuple,
            },
            cross_reducer,
        )


    def _module_update_h(self, G: dgl.heterograph):
        # take equal effect from all task
        update_func_tuple = (fn.copy_u('h', 'm'), fn.max('m', 'h'))

        def cross_reducer(l):
            t = torch.concat(l, dim=-1)
            t = self.link_linear(t)
            t = self.activation_func(t)
            t = self.link_norm(t)
            return t

        G.multi_update_all(
            {
                'use_link': update_func_tuple, # only in link do we care about
            },
            cross_reducer,
        )

    def forward(self, G: dgl.heterograph):
        self._init_inp(G)
        self._task_update_h(G)
        self._module_update_h(G)

class LinkConv(nn.Module):
    """ Links pass messages to each other.
    """
    def __init__(self, 
                 h_dim: int,
                 n_layer: int = 2,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.h_dim = h_dim
        self.n_layer = n_layer
        self.linears = [nn.Linear(2 * h_dim, h_dim) for i in range(self.n_layer)]
        self.activation_func = nn.GELU()

    def _link_message_passing(self, G: dgl.heterograph, timestep: int):
        def cross_reducer(l):
            t = torch.concat(l, dim=-1)
            linear = self.linears[timestep]
            return self.activation_func(linear(t))

        G.multi_update_all(
            {
                'connect_to': (fn.e_mul_u('feat', 'h', 'm'), fn.sum('m', 'h_')),
                'connected_by': (fn.e_mul_u('feat', 'h', 'm'), fn.sum('m', 'h_')),
            },
            cross_reducer,
        )
        G.nodes['link'].data['h'] = G.nodes['link'].data['h'] + G.nodes['link'].data['h_']

    def forward(self, G: dgl.heterograph):
        for i in range(self.n_layer):
            self._link_message_passing(G, i)

class HeteroNet(nn.Module):
    def __init__(self,
                 h_dim: int,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task_info_conv = TaskInfoConv(h_dim, activate_inp=True)
        self.link_conv = LinkConv(h_dim)
        self.head = nn.Linear(h_dim, 1)

    def forward(self, G: dgl.heterograph):
        self.task_info_conv(G)
        self.link_conv(G)
        h = G.nodes['link'].data['h']
        logits = self.head(h)
        logits = torch.sigmoid(logits)
        return logits
    
class NoCeptionNet(nn.Module):
    def __init__(self,
                 h_dim: int,
                 n_layer: int,
                 act_func: str = 'relu',
                 dropout: float = 0.2,
                 *args, 
                 **kwargs,
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.h_dim = h_dim
        assert h_dim % 2 == 0
        self.node_inp_linear = nn.Linear(2, h_dim)
        self.edge_inp_linear = nn.Linear(1, h_dim)

        self.n_layer = n_layer
        self.mi_linears = []
        self.mo_linears = []
        self.node_norms = []
        self.edge_norms = []
        for _ in range(n_layer):
            self.mi_linears.append(nn.Linear(h_dim, h_dim ** 2 // 2))
            self.mo_linears.append(nn.Linear(h_dim, h_dim ** 2 // 2))
            self.node_norms.append(nn.LayerNorm(h_dim))
            self.edge_norms.append(nn.LayerNorm(h_dim))

        if act_func == "relu":
            self.act_func = nn.ReLU()
        elif act_func == 'elu':
            self.act_func = nn.ELU()
        else:
            raise NotImplementedError
        
        self.drop = nn.Dropout(dropout)
        
        # self.gap = dglnn.GlobalAttentionPooling(nn.Linear(h_dim, 1), nn.Linear(h_dim, h_dim))
        self.gap = dglnn.GlobalAttentionPooling(nn.Linear(h_dim, 1))
        self.gap_linear = nn.Linear(h_dim, h_dim)
        self.graph_feat_linear = nn.Linear(1, h_dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            self.act_func,
            nn.Linear(h_dim, 1),
        )

    def forward(self, G: dgl.graph, graph_feat):
        # convert input to hidden
        G.ndata['h'] = self.act_func(self.node_inp_linear(G.ndata['inp']))
        G.edata['h'] = self.act_func(self.edge_inp_linear(G.edata['inp']))

        G_ = dgl.reverse(G, copy_ndata=True, copy_edata=True)

        for l in range(self.n_layer):
            G.ndata['h_'] = self.node_norms[l](G.ndata['h'])
            G_.ndata['h_'] = self.node_norms[l](G_.ndata['h'])

            edge_h = self.edge_norms[l](G.edata['h'])
            ti = self.act_func(self.mi_linears[l](edge_h))
            to = self.act_func(self.mo_linears[l](edge_h))
            ti = self.drop(ti)
            to = self.drop(to)
            G.edata['ti'] = ti.reshape(-1, self.h_dim // 2, self.h_dim)
            G_.edata['to'] = to.reshape(-1, self.h_dim // 2, self.h_dim)
            def mi_func(edges):
                m = edges.data['ti'] @ edges.dst['h_'].unsqueeze(-1)
                m = m.squeeze(-1)
                return {'m': m}
            def mo_func(edges):
                m = edges.data['to'] @ edges.dst['h_'].unsqueeze(-1)
                m = m.squeeze(-1)
                return {'m': m}
            G.update_all(mi_func, fn.sum('m', 'mi'))
            G_.update_all(mo_func, fn.sum('m', 'mo'))

            node_h = G.ndata['h']
            m = torch.concat((G.ndata['mi'], G_.ndata['mo']), dim=-1)
            m = self.drop(m)
            G.ndata['h'] = self.act_func(node_h + m)

        gap_feat = self.gap(G, G.ndata['h']).squeeze(0)
        h0 = self.act_func(self.gap_linear(gap_feat))
        h1 = self.act_func(self.graph_feat_linear(graph_feat))
        h = torch.cat((h0, h1), dim=-1)
        pred = self.final_mlp(h)
        return pred
        