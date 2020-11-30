import logging

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from nasbench301.surrogate_models.gnn.gnn_utils import OP_PRIMITIVES
from nasbench301.surrogate_models.gnn.models.gcn_lib.sparse.torch_nn import norm_layer
from nasbench301.surrogate_models.gnn.models.gcn_lib.sparse.torch_vertex import GENConv


class DeeperGCN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, model_config):
        super(DeeperGCN, self).__init__()

        self.num_layers = model_config['num_gnn_layers']
        self.dropout = model_config['dropout_prob']
        self.block = 'res+'

        hidden_channels = model_config['gnn_hidden_dimensions']
        num_tasks = dim_target
        conv = 'gen'
        aggr = 'add'
        # temperature of softmax
        t = 0.01
        self.learn_t = False
        # the power of power mean
        p = 1.0
        self.learn_p = False
        self.msg_norm = False
        learn_msg_scale = False

        conv_encode_edge = False
        norm = 'layer'
        mlp_layers = 2

        graph_pooling = 'mean'

        print('The number of layers {}'.format(self.num_layers),
              'Aggr aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))
        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.node_features_encoder = torch.nn.Linear(dim_features, hidden_channels)
        self.edge_encoder = torch.nn.Embedding(len(OP_PRIMITIVES), hidden_channels)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception('Unknown Pool Type')

        self.graph_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, graph_batch):

        x = graph_batch.x.float()
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr.long()
        batch = graph_batch.batch

        h = self.node_features_encoder(x)
        edge_emb = self.edge_encoder(edge_attr).squeeze()

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)
                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')

        h_graph = self.pool(h, batch)

        return torch.nn.Sigmoid()(self.graph_pred_linear(h_graph)).view(-1)

    def print_params(self, epoch=None, final=False):
        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))
