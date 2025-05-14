import os
import torch
import numpy as np
import random
from DataLoader import load_data
from paraparser import parameter_parser
from utils import tab_printer
from train import train
from model import Net
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
import torch
import copy
import torch.nn.functional as F
from utils import PairwiseDistance, edge_idx_projection, graph_delete_connections, graph_add_connections, dropout_edge_wrt_sim
from sklearn.decomposition import PCA
from torch_geometric.utils import to_undirected


def ADEdgeDrop(args, seed, id=0):
   
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    tab_printer(args)

    graph = load_data(args)
    graph.edge_index = to_undirected(graph.edge_index)
    number_class = torch.unique(graph.y).shape[0]  
    args.num_class = number_class
    processed_dir = os.path.join(os.path.join(os.path.join("../data",args.dataset_name), args.dataset_name), "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    print(f"Data statistics:  #features {graph.x.size(1)}, #nodes {graph.x.size(0)}, #edges {graph.edge_index.size(1) / 2}")

    sim_path = os.path.join("../dataset", args.dataset_name)
    if not os.path.exists(sim_path):
        os.mkdir(sim_path)
    save_path = os.path.join(sim_path, "sim.pt")
    if os.path.exists(save_path):
        load_sim_dict = torch.load(save_path)
        sim = load_sim_dict["sim"].to(args.device)
        sim_all = load_sim_dict["sim_all"].to(args.device)
        sim_mask = load_sim_dict["sim_mask"].to(args.device)
        print("Similarity matrix loaded.")
    else:
        print("Constructing similarity matrix...")
        sim, sim_all, sim_mask = PairwiseDistance(graph)
        sim = sim.to(args.device)
        sim_all = sim_all.to(args.device)
        sim_mask = sim_mask.to(args.device)
        save_sim_dict = {"sim":sim, "sim_all":sim_all, "sim_mask":sim_mask}
        torch.save(save_sim_dict, save_path)
        print("Similarity matrix constrauction completed.")

    graph = graph.to(device)
    input_channels = graph.x.size(1)
    c = len(torch.unique(graph.y))

    if args.del_edge > 0:
        graph.edge_index, actual_del_prob = graph_delete_connections(args.del_edge, graph.edge_index)
        print('Deleted {}% edges.'.format(100 * actual_del_prob))
        # exit(0)
    elif args.add_edge > 0:
        graph.edge_index, actual_add_prob = graph_add_connections(args.add_edge, graph.edge_index, graph.x.size(0), args.device)
        print('Added {}% edges.'.format(100 * actual_add_prob))

    pca = PCA(n_components = c)
    reduced_X = pca.fit_transform(graph.x.cpu())
    reduced_X = torch.from_numpy(reduced_X).to(device).to(torch.float32)
    edge_feature = F.softmax((torch.concat([reduced_X[graph.edge_index[0], :], reduced_X[graph.edge_index[1], :]], dim=1) + torch.concat([reduced_X[graph.edge_index[1], :], reduced_X[graph.edge_index[0], :]], dim=1) ) / 2, dim=1) 
    graph.edge_attr = edge_feature
    if (graph.is_undirected == False):
        print("The original graph is not an undirected graph!")
        exit(0)
    if args.drop_line_graph > 0:
        lg_masked_edge_index, lg_masked_edge_id = dropout_edge_wrt_sim(graph.edge_index, sim_all, p=args.drop_line_graph) 
        print("Size of line graph:", lg_masked_edge_index.size(1))

        edge_feature = edge_feature[lg_masked_edge_id]


        masked_graph = Data(x=graph.x, edge_index=lg_masked_edge_index, edge_attr = edge_feature).to(device)

        lgraph = LineGraph()(copy.deepcopy(masked_graph))
        lgraph.edge_index = to_undirected(lgraph.edge_index)
        edge_proj = edge_idx_projection(masked_graph)  

        lgraph.edge_proj = lg_masked_edge_id[edge_proj]


        sim_mask_new = torch.isin(sim_mask, lg_masked_edge_id)

        sim = sim[sim_mask_new]
        lg_masked_edge_id = sim_mask_new

        model = Net(args, input_channels, c, sim_all).to(device)
    else:
        lgraph = LineGraph()(copy.deepcopy(graph))
        lgraph.edge_index = to_undirected(lgraph.edge_index)
        edge_proj = edge_idx_projection(graph)  
        lgraph.edge_proj = edge_proj

        model = Net(args, input_channels, c, sim_all).to(device)
        lg_masked_edge_id = None

    if args.dataset_name == "arxiv":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=5e-4)

    test_accuracy_value, incomplete_accuracy_value, edge_num = train(model, optimizer, graph, lgraph, sim, args)
    return test_accuracy_value.cpu(), incomplete_accuracy_value.cpu(), edge_num

if __name__ == "__main__":
    args = parameter_parser()
    accs = []
    accs_incomplete = []
    edge_nums = []
    seed = [args.seed + i for i in range(5)]
    for i in range(5):
        test_accuracy_value, incomplete_accuracy_value, edge_num = ADEdgeDrop(args, seed[i], i)
        accs.append(test_accuracy_value)
        accs_incomplete.append(incomplete_accuracy_value)
        edge_nums.append(edge_num)
    print(f'Avg ACC: {np.mean(accs):.6f}, std: {np.std(accs):.6f}; Avg ACC of incomplete graph: {np.mean(accs_incomplete):.6f}, std: {np.std(accs_incomplete):.6f}; Avg edge numbers: {np.mean(edge_nums):.2f}')

