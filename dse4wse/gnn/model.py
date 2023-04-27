
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
    
class FeatureGen(nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 act_func: nn.Module,
                 dropout: float,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inp_lin = nn.Linear(in_feat, out_feat)
        self.act_func = act_func
        self.mlp = nn.Sequential(
            nn.Linear(out_feat, out_feat),
            act_func,
            nn.Linear(out_feat, out_feat),
        )
        self.norm = nn.LayerNorm(out_feat)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act_func(self.inp_lin(x))
        x_ = self.mlp(x)
        x_ = self.norm(x_)
        x_ = self.drop(x_)
        x = x + x_
        return x
    
class NoCeptionNet(nn.Module):
    def __init__(self,
                 h_dim: int = 64,
                 n_layer: int = 2,
                 act_func: str = 'elu',
                 dropout: float = 0.0,
                 use_norm: bool = True,  # crutial! the more the better hhh
                 use_deeper_mlp_for_inp: bool = False,
                 use_deeper_mlp_for_edge_func: bool = False,
                 use_residual_connect: bool = False,
                 pooling: str = 'max',
                 *args, 
                 **kwargs,
                 ) -> None:
        super().__init__(*args, **kwargs)

        logger.info(f"h_dim: {h_dim}")
        logger.info(f"n_layer: {n_layer}")
        logger.info(f"act_func: {act_func}")
        logger.info(f"dropout: {dropout}")
        logger.info(f"use_norm: {use_norm}")
        logger.info(f"use_deeper_mlp_for_inp: {use_deeper_mlp_for_inp}")
        logger.info(f"use_deeper_mlp_for_edge_func: {use_deeper_mlp_for_edge_func}")
        logger.info(f"use_residual_connect: {use_residual_connect}")
        logger.debug(f"pooling: {pooling}")

        self.h_dim = h_dim
        assert h_dim % 2 == 0

        if act_func == "relu":
            self.act_func = nn.ReLU()
        elif act_func == 'elu':
            self.act_func = nn.ELU()
        else:
            raise NotImplementedError

        node_inp_dim = 13
        edge_inp_dim = 13
        if use_deeper_mlp_for_inp:
            self.node_inp_linear = FeatureGen(node_inp_dim, h_dim, self.act_func, dropout)
            self.edge_inp_linear = FeatureGen(edge_inp_dim, h_dim, self.act_func, dropout)
        else:
            self.node_inp_linear = nn.Sequential(
                nn.Linear(node_inp_dim, h_dim),
                self.act_func,
            )
            self.edge_inp_linear = nn.Sequential(
                nn.Linear(edge_inp_dim, h_dim),
                self.act_func,
            )

        self.n_layer = n_layer
        self.mi_linears = []
        self.mo_linears = []
        self.mi_convs = []
        self.mo_convs = []

        for i in range(n_layer):
            if use_deeper_mlp_for_edge_func:
                self.mi_linears.append(nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    self.act_func,
                    nn.Linear(h_dim, h_dim ** 2 // 2),
                ))
                self.mo_linears.append(nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    self.act_func,
                    nn.Linear(h_dim, h_dim ** 2 // 2),
                ))
            else:
                self.mo_linears.append(nn.Linear(h_dim, h_dim ** 2 // 2))
                self.mi_linears.append(nn.Linear(h_dim, h_dim ** 2 // 2))

            mi_edge_func = self.create_edge_func('in', i)
            mo_edge_func = self.create_edge_func('out', i)

            self.mi_convs.append(dglnn.NNConv(self.h_dim, self.h_dim // 2, mi_edge_func, aggregator_type='max'))
            self.mo_convs.append(dglnn.NNConv(self.h_dim, self.h_dim // 2, mo_edge_func, aggregator_type='max'))

        
        self.use_norm = use_norm
        if use_norm:
            self.node_norms = []
            self.edge_norms = []
            self.message_norms = []
            for i in range(n_layer):
                self.node_norms.append(nn.LayerNorm(h_dim))
                self.edge_norms.append(nn.LayerNorm(h_dim))
                self.message_norms.append(nn.LayerNorm(h_dim))
        
        self.drop = nn.Dropout(dropout)
        self.use_residual_connect = use_residual_connect
        
        if pooling == 'max':
            self.pooling = dglnn.MaxPooling()
        elif pooling == 'set2set':
            self.pooling = dglnn.Set2Set(self.h_dim, 1, 1)
        else:
            raise NotImplementedError
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, 4 * h_dim) if pooling == 'set2set' else nn.Linear(h_dim, 4 * h_dim),
            self.act_func,
            nn.Linear(4 * h_dim, 1),
        )

    def create_edge_func(self, direction, layer):
        assert layer in range(self.n_layer)
        if direction == 'in':
            linear = self.mi_linears[layer]
        elif direction == 'out':
            linear = self.mo_linears[layer]
        else:
            raise NotImplementedError
        return linear

    def forward(self, G: dgl.graph):
        # convert input to hidden
        G.ndata['h'] = self.node_inp_linear(G.ndata['inp'])
        G.edata['h'] = self.edge_inp_linear(G.edata['inp'])
        G.ndata['h'] = self.drop(G.ndata['h'])
        G.edata['h'] = self.drop(G.edata['h'])

        G_ = dgl.reverse(G, copy_ndata=True, copy_edata=True)

        for l in range(self.n_layer):
            nfeat = G.ndata['h'] if not self.use_norm else self.node_norms[l](G.ndata['h'])
            efeat = G.edata['h'] if not self.use_norm else self.node_norms[l](G.edata['h'])
            mi = self.mi_convs[l](G, nfeat, efeat)
            mo = self.mo_convs[l](G_, nfeat, efeat)
            m = torch.concat([mi, mo], dim=-1)
            if self.use_norm:
                m = self.message_norms[l](m)
            if self.use_residual_connect:
                G.ndata['h'] = G.ndata['h'] + self.act_func(m)
            else:
                G.ndata['h'] = self.act_func(nfeat + m)  # NoCeption style

        graph_feat = self.pooling(G, G.ndata['h'])
        pred = self.final_mlp(graph_feat)
        return pred
        