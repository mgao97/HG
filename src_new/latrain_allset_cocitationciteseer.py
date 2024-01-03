#!/usr/bin/env python
# coding: utf-8

import os
import time
# import math
import torch
# import pickle
import argparse
import sys
import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dhg import Hypergraph
from tqdm import tqdm
from tqdm import trange
from layers import *
from models import *
import copy
import torch.optim as optim
from preprocessing import *
import hgnn_cvae_pretrain_allset
from convert_datasets_to_pygDataset import dataset_Hypergraph
from utils import accuracy, normalize_features, micro_f1, macro_f1

import torch
torch.autograd.set_detect_anomaly(True)


def parse_method(args, data):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.concat:
            model = LASetGNN(args,data.norm)
        elif args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                             V=V, E=E)
    #     Below we can add different model, such as HyperGCN and so on
    return model


# class Logger(object):
#     """ Adapted from https://github.com/snap-stanford/ogb/ """

#     def __init__(self, runs, info=None):
#         self.info = info
#         self.results = [[] for _ in range(runs)]

#     def add_result(self, run, result):
#         assert len(result) == 3
#         assert run >= 0 and run < len(self.results)
#         self.results[run].append(result)

#     def print_statistics(self, run=None):
#         if run is not None:
#             result = 100 * torch.tensor(self.results[run])
#             argmax = result[:, 1].argmax().item()
#             print(f'Run {run + 1:02d}:')
#             print(f'Highest Train: {result[:, 0].max():.2f}')
#             print(f'Highest Valid: {result[:, 1].max():.2f}')
#             print(f'  Final Train: {result[argmax, 0]:.2f}')
#             print(f'   Final Test: {result[argmax, 2]:.2f}')
#         else:
#             result = 100 * torch.tensor(self.results)

#             best_results = []
#             for r in result:
#                 train1 = r[:, 0].max().item()
#                 valid = r[:, 1].max().item()
#                 train2 = r[r[:, 1].argmax(), 0].item()
#                 test = r[r[:, 1].argmax(), 2].item()
#                 best_results.append((train1, valid, train2, test))

#             best_result = torch.tensor(best_results)

#             print(f'All runs:')
#             r = best_result[:, 0]
#             print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
#             r = best_result[:, 1]
#             print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
#             r = best_result[:, 2]
#             print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
#             r = best_result[:, 3]
#             print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

#             return best_result[:, 1], best_result[:, 3]

#     def plot_result(self, run=None):
#         plt.style.use('seaborn')
#         if run is not None:
#             result = 100 * torch.tensor(self.results[run])
#             x = torch.arange(result.shape[0])
#             plt.figure()
#             print(f'Run {run + 1:02d}:')
#             plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
#             plt.legend(['Train', 'Valid', 'Test'])
#         else:
#             result = 100 * torch.tensor(self.results[0])
#             x = torch.arange(result.shape[0])
#             plt.figure()
# #             print(f'Run {run + 1:02d}:')
#             plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
#             plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Main part of the training ---
# # Part 0: Parse arguments


def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = hgnn_cvae_pretrain_allset.feature_tensor_normalize(augmented_features).detach()
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list

def get_hyperedges_from_incident_matrix(incident_matrix):
        num_nodes, num_hyperedges = incident_matrix.shape
        hyperedges_list = []

        for hyperedge_idx in range(num_hyperedges):
            nodes_in_hyperedge = np.where(incident_matrix[:, hyperedge_idx] == 1)[0].tolist()
            hyperedges_list.append(nodes_in_hyperedge)

        return hyperedges_list



# Adapted from GRAND: https://github.com/THUDM/GRAND
def consis_loss(logps, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss

"""

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--concat", type=int, default=4)
    parser.add_argument("--latent_size", type=int, default=10)
    parser.add_argument('--dataset', default='citeseer', help='Dataset string.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
    parser.add_argument('--lam', type=float, default=1., help='Lamda')
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--pretrain_lr", type=float, default=0.001)
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
    parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
    parser.add_argument('--warmup', type=int, default=200, help='Warmup')

    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='citeseer')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='AllDeepSets')
    # parser.add_argument('--epochs', default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=3, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    # parser.add_argument('--dropout', default=0.5, type=float)
    # parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=8,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=6, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    #     Args for Attentions: GAT and SetGNN
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    
    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()
    
    print('args\n', args)
    
    # # Part 1: Load data
    
    
    ### Load and preprocess data ###
    existing_dataset = ['contact-high-school','20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed']
    
    synthetic_list = ['contact-high-school','amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']
    
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = '../data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = '../data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        data = dataset.data
        # print('data.x:',data.x.shape)
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ['contact-high-school','yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
    
    # ipdb.set_trace()
    #     Preprocessing
    # if args.method in ['SetGNN', 'SetGPRGNN', 'SetGNN-DeepSet']:
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)
    
        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')
    
    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)
    #     ipdb.set_trace()
    #   Feature normalization, default option in HyperGCN
        # X = data.x
        # X = sp.csr_matrix(utils.normalise(np.array(X)), dtype=np.float32)
        # X = torch.FloatTensor(np.array(X.todense()))
        # data.x = X
    
    # elif args.method in ['HGNN']:
    #     data = ExtractV2E(data)
    #     if args.add_self_loop:
    #         data = Add_Self_Loops(data)
    #     data = ConstructH(data)
    #     data = generate_G_from_H(data)
    
    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
    
    elif args.method in ['HCHA', 'HGNN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
    #    Make the first he_id to be 0
        data.edge_index[1] -= data.edge_index[1].min()
        
    elif args.method in ['UniGCNII']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
        # Compute degV and degE
        if args.cuda in [0,1]:
            device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        (row, col), value = torch_sparse.from_scipy(data.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)

        degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
        from torch_scatter import scatter
        degE = scatter(degV[V], E, dim=0, reduce='mean')
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1
        args.UniGNN_degV = degV
        args.UniGNN_degE = degE
    
        V, E = V.cpu(), E.cpu()
        del V
        del E
    
    
    
    H = ConstructH(data)
    he_list = get_hyperedges_from_incident_matrix(H)
    he_list = [tuple(i) for i in he_list]
    
    hg = Hypergraph(data.x.shape[0], he_list)
    labels = data.y
    features = data.x

    features_normalized = normalize_features(features.numpy())
    labels = data.y
    features_normalized = torch.FloatTensor(features_normalized)
    #     Get splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
    
    print('data info:', data)
    # # Part 2: Load model
    
    model = parse_method(args, data)
    print('model:\n',model)
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    model, data = model.to(device), data.to(device)
    hg = hg.to(device)
    labels = labels.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    split_idx = rand_train_test_idx(
        data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
    
    idx_train, idx_val, idx_test = split_idx['train'], split_idx['valid'], split_idx['test']
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)
    
    num_params = count_parameters(model)
    
    exc_path = sys.path[0]
    cvae_model = torch.load("{}/model/{}_0102.pkl".format(exc_path, args.dataset))
    # # Part 3: Main. Training + Evaluation
    
    if args.cuda:
        hg = hg.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        features_normalized = features_normalized.to(device)

    all_val = []
    all_test = []
    for i in trange(args.runs, desc='Run Train'):


        # Train model
        best = 999999999
        best_model = None
        best_X_list = None
        for epoch in range(args.epochs):

            model.train()
            optimizer.zero_grad()

            output_list = []
            for k in range(args.samples):
                X_list = get_augmented_features(args.concat)
                # print('len(X_list), X_list[0].shape:', len(X_list), X_list[0].shape)
                # print('features_normalized:',features_normalized.shape)
                output_list.append(torch.log_softmax(model(X_list+[features_normalized], data), dim=-1))

            loss_train = 0.
            for k in range(len(output_list)):
                loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
            
            loss_train = loss_train/len(output_list)

            loss_consis = consis_loss(output_list,temp=args.tem)
            loss_train = loss_train + loss_consis

            loss_train.backward()
            optimizer.step()

            model.eval()
            val_X_list = get_augmented_features(args.concat)
            output = model(val_X_list+[features_normalized],data)
            output = torch.log_softmax(output, dim=1)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            

            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()))
                    
            if loss_val < best:
                best = loss_val
                best_model = copy.deepcopy(model)
                best_X_list = copy.deepcopy(val_X_list)

        #Validate and Test
        best_model.eval()
        output = best_model(best_X_list+[features_normalized], data)
        output = torch.log_softmax(output, dim=1)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        all_val.append(acc_val.item())
        all_test.append(acc_test.item())

    print(np.mean(all_val), np.std(all_val), np.mean(all_test), np.std(all_test))
    quit()
    
