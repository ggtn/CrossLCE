from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
from ..layers import EmbeddingLayer, MPLayer, InterConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple


class Tensor_GGNN_GCN(MessagePassing):
    def __init__(self,
                 num_edge_types,
                 #in_features,
                 out_features,
                 embedding_features,
                 #classifier_features,
                 #embedding_num_classes,
                 dropout=0,
                 max_node_per_graph=600,
                 #max_variable_candidates=5,
                 add_self_loops=False,
                 bias=True,
                 aggr="mean",
                 device="cpu",
                 output_model="learning"):
        super(Tensor_GGNN_GCN, self).__init__(aggr=aggr)
        # params set
        self.num_edge_types = num_edge_types
        self.device = device
        self.output_model = output_model.lower()
        self.dropout = dropout
        self.max_node_per_graph=max_node_per_graph
        #self.max_variable_candidates = max_variable_candidates
        # 先对值进行embedding
        #self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                   #in_features,
                                                   #embedding_features,
                                                   #device=device)

        self.MessagePassingNN = nn.ModuleList(
            [
                 MPLayer(in_features=embedding_features,
                    out_features=out_features,device=device) for _ in range(self.num_edge_types)
            ]
        )

        self.gru_cell = torch.nn.GRUCell(input_size=embedding_features, hidden_size=out_features)
        
        self.conv1_list = nn.ModuleList([GCNConv(out_features, out_features, add_self_loops=add_self_loops) for _ in range(self.num_edge_types)] )
        self.interConv = InterConv(out_features, out_features, num_edge_types=self.num_edge_types)
        self.thirdNorm = torch.nn.InstanceNorm2d(out_features)  # 这个就是自己要找的

        self.lin = nn.Linear(out_features*2, out_features)
        self.conv1 = GCNConv(out_features, out_features, add_self_loops=add_self_loops)
        self.conv2 = GCNConv(out_features, out_features, add_self_loops=add_self_loops)
        #self.varmisuse_output_layer = nn.Linear(out_features* 2 + 1, 1)
        #self.varnaming_output_layer = nn.Linear(out_features, classifier_features)

    def forward(self, x1, edge_list1: List[torch.tensor], x2, edge_list2: List[torch.tensor]):
        
        x_embedding = x1
        #x_embedding = self.value_embeddingLayer(x)
        # Tensor GGNN 
        last_node_states = x_embedding
        for _ in range(4):
            out_list = []
            cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
            for i in range(len(edge_list1)):
                edge = edge_list1[i]
                if edge.shape[0] != 0 :
                    # 该种类型的边存在边
                    out_list.append(self.MessagePassingNN[i](cur_node_states, edge))
            cur_node_states = sum(out_list)
            new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
            last_node_states = new_node_states

        ggnn_out1 = last_node_states # shape: V, D

        # tensor GCN new:
        assert self.num_edge_types == 4   #4
        cur_x = torch.cat([ggnn_out1,ggnn_out1,ggnn_out1,ggnn_out1], dim=0)  # 4V, D
        loop_edge_list = self.matrix_loop(edge_list1) # 4V, 4V
        out = self.conv1(cur_x, loop_edge_list) # 4V, D
        out = self.conv2(out, loop_edge_list)  # 4V, D
        out = out.view( 4, x_embedding.shape[0], out.shape[-1])  # V, 4, D
        out = torch.sum(out, dim=0)  # V, D
        n1 = out
  #.............................................................................................

        x_embedding = x2
        #x_embedding = self.value_embeddingLayer(x)
        # Tensor GGNN
        last_node_states = x_embedding
        for _ in range(4):
            out_list = []
            cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
            for i in range(len(edge_list2)):
                edge = edge_list2[i]
                if edge.shape[0] != 0 :
                    # 该种类型的边存在边
                    out_list.append(self.MessagePassingNN[i](cur_node_states, edge))
            cur_node_states = sum(out_list)
            new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
            last_node_states = new_node_states

        ggnn_out2 = last_node_states # shape: V, D

        # tensor GCN new:
        assert self.num_edge_types == 4   #4
        cur_x = torch.cat([ggnn_out2,ggnn_out2,ggnn_out2,ggnn_out2], dim=0)  # 4V, D
        loop_edge_list = self.matrix_loop(edge_list2) # 4V, 4V
        out = self.conv1(cur_x, loop_edge_list) # 4V, D
        out = self.conv2(out, loop_edge_list)  # 4V, D
        out = out.view( 4, x_embedding.shape[0], out.shape[-1])  # V, 4, D
        out = torch.sum(out, dim=0)  # V, D
        n2 = out

   #............................................................................

        cos = torch.cosine_similarity(n1, n2, dim=0)

        return cos



    def matrix_transfer(self, edge, i, j):
        # edge: [[i],[j]]
        edge_new = edge.detach().clone()
        edge_new[0]+=i
        edge_new[1]+=j
        return edge_new

    def matrix_loop(self, edge_list):
        # 4个邻接矩阵并列。
        assert len(edge_list) == 4
        A1, A2, A3, A4 = edge_list
        n = self.max_node_per_graph
        loop_edge_list = []
        loop_edge_list.append(A1)
        loop_edge_list.append(self.matrix_transfer(A2, n, 0))
        loop_edge_list.append(self.matrix_transfer(A3, 2*n, 0))
        loop_edge_list.append(self.matrix_transfer(A4, 3*n, 0))

        loop_edge_list.append(self.matrix_transfer(A4, 0, n))
        loop_edge_list.append(self.matrix_transfer(A1, n, n))
        loop_edge_list.append(self.matrix_transfer(A2, 2*n, n))
        loop_edge_list.append(self.matrix_transfer(A3, 3*n, n))

        loop_edge_list.append(self.matrix_transfer(A3, 0, 2*n))
        loop_edge_list.append(self.matrix_transfer(A4, n, 2*n))
        loop_edge_list.append(self.matrix_transfer(A1, 2*n, 2*n))
        loop_edge_list.append(self.matrix_transfer(A2, 3*n, 2*n))

        loop_edge_list.append(self.matrix_transfer(A2, 0, 3*n))
        loop_edge_list.append(self.matrix_transfer(A3, n, 3*n))
        loop_edge_list.append(self.matrix_transfer(A4, 2*n, 3*n))
        loop_edge_list.append(self.matrix_transfer(A1, 3*n, 3*n))

        return torch.cat(loop_edge_list, dim=1)

    # def variable_detection(self, out):
    #
    #
    #     return self.out
