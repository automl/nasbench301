import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from nasbench301.surrogate_models.gnn.gnn_utils import NODE_PRIMITIVES
from nasbench301.surrogate_models.gnn.models.conv import GNN_node_Virtualnode, GNN_node


class NodeEncoder(torch.nn.Module):
    '''
    Input:
        x: default node feature. the first and second column represents node type and node attributes.
        depth: The depth of the node in the AST.

    Output:
        emb_dim-dimensional vector

    '''

    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes):
        super(NodeEncoder, self).__init__()

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)

    def forward(self, x):
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(x[:, 1])


class GIN(torch.nn.Module):

    def __init__(self, dim_features, dim_target, model_config, JK="last"):
        super(GIN, self).__init__()

        self.config = model_config
        self.node_encoder = NodeEncoder(model_config['gnn_hidden_dimensions'], num_nodetypes=len(NODE_PRIMITIVES),
                                        num_nodeattributes=8)

        if model_config['virtual_node']:
            self.gnn_node = GNN_node_Virtualnode(num_layer=model_config['num_gnn_layers'],
                                                 emb_dim=model_config['gnn_hidden_dimensions'],
                                                 JK=JK, drop_ratio=model_config['dropout_prob'], residual=False,
                                                 gnn_type='gin', node_encoder=self.node_encoder)
        else:
            self.gnn_node = GNN_node(num_layer=model_config['num_gnn_layers'],
                                     emb_dim=model_config['gnn_hidden_dimensions'],
                                     JK=JK, drop_ratio=model_config['drop_ratio'], residual=False,
                                     gnn_type='gin', node_encoder=self.node_encoder)
        if model_config['graph_pooling'] == "sum":
            self.pool = global_add_pool
        elif model_config['graph_pooling'] == "mean":
            self.pool = global_mean_pool
        elif model_config['graph_pooling'] == "max":
            self.pool = global_max_pool
        elif model_config['graph_pooling'] == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(model_config['gnn_hidden_dimensions'], 2 * model_config['gnn_hidden_dimensions']),
                    torch.nn.BatchNorm1d(2 * model_config['gnn_hidden_dimensions']),
                    torch.nn.ReLU(), torch.nn.Linear(2 * model_config['gnn_hidden_dimensions'], 1)))
        elif model_config['graph_pooling'] == "set2set":
            self.pool = Set2Set(model_config['gnn_hidden_dimensions'], processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear_list = torch.nn.ModuleList()

        self.graph_pred_linear = torch.nn.Linear(model_config['gnn_hidden_dimensions'], 1)

    def forward(self, graph_batch):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer

        h_node = self.gnn_node(graph_batch)

        h_graph = self.pool(h_node, graph_batch.batch)
        graph_output = self.graph_pred_linear(h_graph)
        return torch.sigmoid(graph_output.view(-1))
