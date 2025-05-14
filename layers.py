import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, args, dim_in, dim_out, EdgePredictor = False):
        super(GCN, self).__init__()
        self.args = args
        self.dropout_rate = args.dropout

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.relu = nn.ReLU(inplace=True)

        
        self.EdgePredictor = EdgePredictor

        if self.EdgePredictor == True:
            self.layer_num = 2
        else:
            self.layer_num = args.layer_num

        self.setup_layers(args)

    def forward(self, x, edge_index):
        z = x
        for i in range(self.layer_num - 1):
            z = self.relu(self.layers[i](x = z, edge_index=edge_index))
            z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = self.layers[-1](x = z, edge_index=edge_index)
        
        return z
    
    def setup_layers(self, args):
        self.args.layers = [args.hidden_dim for i in range(self.layer_num - 1)]
        self.args.layers = [self.dim_in] + self.args.layers + [self.dim_out]

        self.layers = nn.ModuleList()

        for i, _ in enumerate(self.args.layers[:-1]):
            self.layers.append(GCNConv(self.args.layers[i], self.args.layers[i+1]))


class SAGE(nn.Module):
    def __init__(self, args, dim_in, dim_out, EdgePredictor = False):
        super(SAGE, self).__init__()
        self.args = args
        self.dropout_rate = args.dropout

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.relu = nn.ReLU(inplace=True)

        
        self.EdgePredictor = EdgePredictor

        if self.EdgePredictor == True:
            self.layer_num = 2
        else:
            self.layer_num = args.layer_num

        self.setup_layers(args)

    def forward(self, x, edge_index):
        z = x
        for i in range(self.layer_num - 1):
            z = self.relu(self.layers[i](x = z, edge_index=edge_index))
            z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = self.layers[-1](x = z, edge_index=edge_index)
        
        return z
    
    def setup_layers(self, args):
        """
        Creating the layes based on the args.
        """
        # set layer weights for GCN
        self.args.layers = [args.hidden_dim for i in range(self.layer_num - 1)]
        self.args.layers = [self.dim_in] + self.args.layers + [self.dim_out]

        self.layers = nn.ModuleList()

        for i, _ in enumerate(self.args.layers[:-1]):
            self.layers.append(SAGEConv(self.args.layers[i], self.args.layers[i+1]))


class GAT(nn.Module):
    def __init__(self, args, dim_in, dim_out, EdgePredictor = False):
        super(GAT, self).__init__()
        self.args = args
        self.dropout_rate = args.dropout

        self.dim_in = dim_in
        self.dim_out = dim_out

        
        self.EdgePredictor = EdgePredictor

        if self.EdgePredictor == True:
            self.layer_num = 2
        else:
            self.layer_num = args.layer_num

        self.setup_layers(args)

    def forward(self, x, edge_index):
        z = x
        for i in range(self.layer_num - 1):
            z = F.elu(self.layers[i](x = z, edge_index=edge_index))
            z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = self.layers[-1](x = z, edge_index=edge_index)
        
        return z
    
    def setup_layers(self, args):
        self.args.layers = [args.hidden_dim for i in range(self.layer_num - 1)]
        self.args.layers = [self.dim_in] + self.args.layers + [self.dim_out]

        self.layers = nn.ModuleList()

        for i, _ in enumerate(self.args.layers[:-1]):
            self.layers.append(GATConv(self.args.layers[i], self.args.layers[i+1]))



class EdgePredictorLayer(nn.Module):
    def __init__(self, args, edge_attr_dim, backbone):
        super(EdgePredictorLayer, self).__init__()
        
        self.dim_in = edge_attr_dim
        self.dim_out = 2 

        self.backbone = backbone

        if self.backbone == "GCN":
            self.backbone = GCN(args, self.dim_in, self.dim_out, EdgePredictor = True)
        elif self.backbone == "SAGE":
            self.backbone = SAGE(args, self.dim_in, self.dim_out)
        elif self.backbone == "GAT":
            self.backbone = GAT(args, self.dim_in, self.dim_out)

    def forward(self, lgraph):
        x = lgraph.x
        edge_index = lgraph.edge_index
        z = self.backbone(x, edge_index)
            
        return z

