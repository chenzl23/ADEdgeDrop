import torch
import copy
import torch.nn.functional as F
from utils import accuracy
import time


def edge_prediction_loss_binary(edge_emb, sim, thred):
    gnd_sim = (sim > thred).long()  
    loss = - torch.log(edge_emb[gnd_sim, 1]).mean()
    return loss

def train(model, optimizer, graph, lgraph, sim, args):
    best_valid_acc = 0.0
    patience = args.patience
    best_model = copy.deepcopy(model)
    best_epoch = 0

    best_lgraph = None
    best_masked_edge_index = None

    loss_list = []
    acc_valid_list = []
    acc_test_list = []

    for epoch in range(1, args.epoch_num + 1):
        start_time = time.time()
        loss, valid_acc, model, optimizer, new_edge_feature, masked_edge_index = train_fullbatch(model, optimizer, graph, graph.edge_index, lgraph, sim, args)
        if args.verbose == 1:
            print("Epoch: {0:d}".format(epoch), 
                "Training loss: {0:1.5f}".format(loss.cpu().detach().numpy()), 
                "Valid accuracy: {0:1.5f}".format(valid_acc),
                "Time: {0:.3f}".format(time.time() - start_time)
                )
        lgraph.x = args.alpha * lgraph.x + (1 - args.alpha) * new_edge_feature.detach()
        if (valid_acc >= best_valid_acc):
                best_valid_acc = valid_acc
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                best_lgraph = copy.deepcopy(lgraph)
                best_masked_edge_index = masked_edge_index
        if args.early_stop:
            if (valid_acc >= best_valid_acc):
                patience = args.patience
            else:
                patience -= 1
                if (patience < 0):
                    print("Early Stopped!")
                    break
        acc_valid_list.append(valid_acc.cpu().detach().numpy())
        loss_list.append(loss.cpu().detach().numpy())
        with torch.no_grad():
            model.eval()
            Z, _, cur_masked_edge_index = model(graph.x, graph.edge_index, best_lgraph)
            predictions = F.log_softmax(Z, dim=1)
            test_accuracy_value = accuracy(predictions[graph.test_mask], graph.y[graph.test_mask])
            acc_test_list.append(test_accuracy_value.cpu().detach().numpy())
            
    test_model = best_model
    with torch.no_grad():
        test_model.eval()
        Z, _, cur_masked_edge_index = test_model(graph.x, graph.edge_index, best_lgraph)
        predictions = F.log_softmax(Z, dim=1)
        test_accuracy_value = accuracy(predictions[graph.test_mask], graph.y[graph.test_mask])
    print("Best epoch:", str(best_epoch))
    print("Test accuracy: {0:1.5f}".format(test_accuracy_value))
    with torch.no_grad():
        test_model.eval()
        Z, _, cur_masked_edge_index = test_model(graph.x, best_masked_edge_index, best_lgraph)
        predictions = F.log_softmax(Z, dim=1)
        accuracy_value = accuracy(predictions[graph.test_mask], graph.y[graph.test_mask])
    edge_num = best_masked_edge_index.size(1)
    return test_accuracy_value, accuracy_value, edge_num


def train_fullbatch(model, optimizer, graph, edge_index, lgraph, sim, args):
    model.train()
    optimizer.zero_grad()
    perturb = torch.FloatTensor(lgraph.x.size(0), 2).uniform_(-args.gamma, args.gamma).to(args.device) 
    perturb.requires_grad_()
    Z, edge_emb, masked_edge_index = model(graph.x, edge_index, lgraph, perturb)

    edge_emb = F.softmax(edge_emb, dim=1)
    loss_ep = edge_prediction_loss_binary(edge_emb, sim, args.thred)
    loss_ep /= args.m
    for _ in range(args.m-1):
        loss_ep.backward()
        perturb_data = perturb.detach() +  args.gamma * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0
        Z, edge_emb, masked_edge_index = model(graph.x, edge_index, lgraph, perturb)

        edge_emb = F.softmax(edge_emb, dim=1)
        loss_ep = edge_prediction_loss_binary(edge_emb, sim, args.thred)
        loss_ep /= args.m
    loss_ep.backward()

    new_edge_feature = F.softmax(torch.concat([Z[edge_index[0, lgraph.edge_proj], :], Z[edge_index[1, lgraph.edge_proj], :]], dim=1), dim=1)
    predictions = F.log_softmax(Z, dim=1)

    loss_ce = F.nll_loss(predictions[graph.train_mask], graph.y[graph.train_mask])
    loss_ce.backward()
    loss = loss_ce + loss_ep 

    optimizer.step()


    with torch.no_grad():
        model.eval()
        Z, _, _ = model(graph.x, edge_index, lgraph)  
        predictions = F.log_softmax(Z, dim=1)
        valid_acc = accuracy(predictions[graph.valid_mask], graph.y[graph.valid_mask])
    
    return loss, valid_acc, model, optimizer, new_edge_feature, masked_edge_index