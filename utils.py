from texttable import Texttable
import torch
from torch_geometric.utils import remove_self_loops, coalesce, to_undirected
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def PairwiseDistance(data):
    N = data.num_nodes
    x = data.x
    edge_index = data.edge_index
    edge_index = coalesce(edge_index, num_nodes=N)
    row, col = edge_index
    mask = torch.where(row < col)[0]

    num_edge = edge_index.size(1)
    sim_all = torch.zeros(num_edge)
    with tqdm(total=num_edge, desc="Constructing similarities") as pbar:
        for i in range(num_edge):
            sim_all[i] = cosine_similarity(x[edge_index[0,i]].unsqueeze(0), x[edge_index[1,i]].unsqueeze(0))
            pbar.update(1)
    sim = sim_all[mask]

    return sim, sim_all, mask

def edge_idx_projection(data):
    edge_index, edge_attr = data.edge_index, data.edge_attr
    row, col = edge_index
    mask = torch.where(row < col)[0]
    return mask

def graph_delete_connections(prob_del, edge_index):
    pre_num_edges = edge_index.size(1)
    row, col = edge_index
    mask = torch.where(row < col)[0]
    edge_index = edge_index[:, mask]  

    remain_mask = torch.rand(edge_index.size(1)) > prob_del

    del_edge_index = edge_index[:, remain_mask]
    del_edge_index = to_undirected(del_edge_index)

    new_edge_num = del_edge_index.size(1)
    actual_del_prob = (pre_num_edges - new_edge_num) / pre_num_edges

    return del_edge_index, actual_del_prob


def graph_add_connections(prob_add, edge_index, N, device):
    pre_num_edges = edge_index.size(1)
    row, col = edge_index
    mask = torch.where(row < col)[0]
    edge_index = edge_index[:, mask]  

    add_edge_index = torch.randint(N, (2, int(edge_index.size(1) * prob_add))).to(device)
    add_edge_index = torch.cat([edge_index, add_edge_index], dim=-1)
    
    add_edge_index, _ = remove_self_loops(add_edge_index)
    add_edge_index, _ = coalesce(add_edge_index, None, N, N)

    add_edge_index = to_undirected(add_edge_index)

    new_edge_num = add_edge_index.size(1)
    actual_add_prob = (new_edge_num - pre_num_edges) / pre_num_edges

    return add_edge_index, actual_add_prob

def dropout_edge_wrt_sim(edge_index, sim_all, p: float = 0.9):
    edge_mask = torch.where((sim_all > p) == True)[0]  
    edge_index = edge_index[:, edge_mask]
    return edge_index, edge_mask

