import torch, os
from torch_geometric import utils
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import scipy.io
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import normalize

def load_data(args):
    data_name = args.dataset_name
    if data_name in ['pubmed']:
        return load_ptg_data(args)
    elif data_name in ['arxiv']:
        return load_ogb_data(args)
    else:
        return load_classic_data(args)

def load_ptg_data(args):
    DATA_ROOT = './dataset'
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    dataset = Planetoid(os.path.join(DATA_ROOT, args.dataset_name), args.dataset_name, num_train_per_class=args.num_train_per_class, num_val=args.num_val, num_test=args.num_test)
    graph = dataset[0]

    graph.valid_mask = graph.val_mask

    if args.feature_normalize == 1:
        print("Feature Normalized.")
        graph.x = torch.from_numpy(normalize(graph.x)).float()
    
    return graph

def load_ogb_data(args, downsampling=1):
    dataset = PygNodePropPredDataset(name='ogbn-'+ args.dataset_name) 
    graph = dataset[0]
    split_idx = dataset.get_idx_split()
    for key, idx in split_idx.items():
        if key=='train' and downsampling < 1:
            perm = torch.randperm(idx.size(0))
            k = int(len(idx) * downsampling)
            idx = idx[perm[:k]]
        mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask[idx] = True
        graph[f'{key}_mask'] = mask
    
    graph.y = graph.y.squeeze()
    if args.feature_normalize == 1:
        print("Feature Normalized.")
        graph.x = torch.from_numpy(normalize(graph.x)).float()
    return graph


def load_classic_data(args):
    data_path = os.path.join("./dataset", args.dataset_name)

    features, edge_index, gnd, train_mask, valid_mask, test_mask = loadMatData(data_path, args)

    data = Data(x=features, edge_index = edge_index, y=gnd, train_mask = train_mask, valid_mask = valid_mask, test_mask = test_mask)

    return data

def count_each_class_num(gnd):
    count_dict = {}
    for label in gnd:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict



def generate_permutation(gnd, args):
    N = gnd.shape[0]
    each_class_num = count_each_class_num(gnd)
    training_each_class_num = {} 

    for label in each_class_num.keys():
        if args.dataset_name in ['CoraFull']:
            training_each_class_num[label] = round(each_class_num[label] * 0.07073)
        else:
            training_each_class_num[label] = args.num_train_per_class
    valid_num = args.num_val
    test_num = args.num_test

    train_mask = torch.from_numpy(np.full((N), False))
    valid_mask = torch.from_numpy(np.full((N), False))
    test_mask = torch.from_numpy(np.full((N), False))

    data_idx = np.random.permutation(range(N))

    for idx in data_idx:
        label = gnd[idx]
        if (training_each_class_num[label] > 0):
            training_each_class_num[label] -= 1
            train_mask[idx] = True
    for idx in data_idx:
        if train_mask[idx] == True:
            continue
        if (valid_num > 0):
            valid_num  -= 1
            valid_mask[idx] = True
        elif (test_num > 0):
            test_num -= 1
            test_mask[idx] = True

    data_dict = {
    'train_idx': torch.where(train_mask == True)[0].numpy(),
    'valid_idx': torch.where(valid_mask == True)[0].numpy(),
    'test_idx': torch.where(test_mask == True)[0].numpy()}
    save_path = os.path.join("./data", args.dataset_name)
    os.mkdir(save_path)
    file_path = os.path.join(save_path, 'data.mat')
    scipy.io.savemat(file_path, data_dict)
    return train_mask, valid_mask, test_mask

def loadMatData(data_path, args):
    data = scipy.io.loadmat(data_path) 
    features = data['X']
    features = torch.from_numpy(features).float()


    gnd = data['Y']
    gnd = gnd.flatten()
    if np.min(gnd) == 1:
        gnd = gnd - 1
    gnd = torch.from_numpy(gnd)

    adj = data['adj']
    adj = torch.from_numpy(adj)

    mask_save_path = os.path.join("./data", args.dataset_name)
    if os.path.exists(mask_save_path):
        mask_data = scipy.io.loadmat(os.path.join(mask_save_path, 'data.mat')) 
        train_mask = mask_data['train_idx']
        valid_mask = mask_data['valid_idx']
        test_mask = mask_data['test_idx']
    else:
        train_mask, valid_mask, test_mask = generate_permutation(gnd.numpy(), args)


    adj = adj + adj.t().multiply(adj.t() > adj) - adj.multiply(adj.t() > adj)
    adj = adj + torch.eye(adj.shape[0], adj.shape[0])
    edge_index = np.argwhere(adj > 0)

    edge_index = utils.remove_self_loops(edge_index)[0]

    if args.feature_normalize == 1:
        print("Feature Normalized.")
        features = torch.from_numpy(normalize(features)).float()
        
    return features, edge_index, gnd, np.squeeze(train_mask), np.squeeze(valid_mask), np.squeeze(test_mask)


