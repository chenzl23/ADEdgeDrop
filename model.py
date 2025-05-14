import torch.nn as nn
import torch.nn.functional as F
from layers import EdgePredictorLayer, GCN, SAGE, GAT
import torch
from torch_geometric.utils import to_undirected,dropout_edge

class Net(nn.Module):
    def __init__(self, args, dim_in, dim_out, sim_all, lg_masked_edge_id = None):
        super(Net, self).__init__()
        self.args = args
        self.dropout_rate = args.dropout

        self.dim_in = dim_in
        self.dim_out = dim_out

        if args.backbone == "GCN":
            self.backbone = GCN(args, dim_in, dim_out)
        elif args.backbone == "SAGE":
            self.backbone = SAGE(args, dim_in, dim_out)
        elif args.backbone == "GAT":
            self.backbone = GAT(args, dim_in, dim_out)

        self.EdgePredictor = EdgePredictorLayer(args, dim_out * 2, args.backbone)

        self.drop_line_graph = args.drop_line_graph

        self.lg_masked_edge_id = lg_masked_edge_id

        self.sigmoid = nn.Sigmoid()

        self.sim_all = sim_all

        self.thred = args.thred


    def forward(self, x, edge_index, lgraph, perturb = None):
        if self.training:
            edge_emb = F.softmax(self.EdgePredictor(lgraph), dim=1) + perturb  
            normalized_edge_emb = edge_emb[:, 1] 
            
            if self.drop_line_graph > 0:
                edge_retain_index = torch.where(normalized_edge_emb > self.thred)[0] 
                edge_retain_index_ori = lgraph.edge_proj[edge_retain_index] 
                edge_mask = torch.zeros(edge_index.size(1),dtype=torch.bool,device=edge_index.device)
                edge_mask[edge_retain_index_ori] = True
                _, random_edge_drop_id = dropout_edge(edge_index, p=0.5)
                edge_mask[random_edge_drop_id] = True
                masked_edge_index = edge_index[:, edge_mask]
                masked_edge_index = to_undirected(masked_edge_index)
            else:
                edge_retain_index = torch.where(normalized_edge_emb > self.thred)[0]

                edge_retain_index_ori = lgraph.edge_proj[edge_retain_index] 
                masked_edge_index = edge_index[:, edge_retain_index_ori]
                masked_edge_index = to_undirected(masked_edge_index)
            z = self.backbone(x, masked_edge_index)
        else:
            edge_emb = None
            masked_edge_index = edge_index
            z = self.backbone(x, masked_edge_index)

        return z, edge_emb, masked_edge_index
    