import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import hgnn_cvae_pretrain_allset

# from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
# from gcn.models import GCN
# from hgnn_cvae_pretrain import HGNN
from tqdm import trange
import dhg
# from dhg.data import CocitationCora, Cooking200
from dhg import Hypergraph
from dhg.nn import HGNNConv
from sklearn.model_selection import train_test_split
from train_allset import *

import os
import time
# import math
import torch
# import pickle
import argparse

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from layers import *
from models import *
from preprocessing import *


from layers import *
from models import *
from preprocessing import *
from convert_datasets_to_pygDataset import dataset_Hypergraph

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
print('generate parsers:\n')
parser.add_argument("--pretrain_epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=0.05)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

parser.add_argument('--dataset', default='coauthor_cora',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

print('allset parsers:\n')
parser.add_argument('--train_prop', type=float, default=0.5)
parser.add_argument('--valid_prop', type=float, default=0.25)
parser.add_argument('--dname', default='coauthor_cora')
# method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
parser.add_argument('--method', default='AllDeepSets')
# parser.add_argument('--epochs', default=500, type=int)
# Number of runs for each split (test fix, only shuffle train/val)
# parser.add_argument('--runs', default=20, type=int)
parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
# parser.add_argument('--dropout', default=0.5, type=float)
# parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=0.0, type=float)
# How many layers of full NLConvs
parser.add_argument('--All_num_layers', default=2, type=int)
parser.add_argument('--MLP_num_layers', default=2,
                    type=int)  # How many layers of encoder
parser.add_argument('--MLP_hidden', default=512,
                    type=int)  # Encoder hidden units
parser.add_argument('--Classifier_num_layers', default=7,
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
parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
# Choose std for synthetic feature noise
parser.add_argument('--feature_noise', default='1', type=str)
# whether the he contain self node or not
parser.add_argument('--exclude_self', action='store_true')
parser.add_argument('--PMA', action='store_true')
#     Args for HyperGCN
parser.add_argument('--HyperGCN_mediators', action='store_true')
parser.add_argument('--HyperGCN_fast', action='store_true')
#     Args for Attentions: GAT and SetGNN
parser.add_argument('--heads', default=4, type=int)  # Placeholder
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

args = parser.parse_args()


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

print('args:\n', args)

# Load data
# adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
        if args.LearnMask:
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
    print('data 0:',data)
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
    # print('data 1:',data)
# ipdb.set_trace()
#     Preprocessing
# if args.method in ['SetGNN', 'SetGPRGNN', 'SetGNN-DeepSet']:
if args.method in ['AllSetTransformer', 'AllDeepSets']:
    # print('data 2:',data)
    data = ExtractV2E(data)
    # print('data 3:',data)
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

# #     Get splits
# split_idx_lst = []
# for run in range(args.runs):
#     split_idx = rand_train_test_idx(
#         data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
#     split_idx_lst.append(split_idx)

def get_hyperedges_from_incident_matrix(incident_matrix):
    num_nodes, num_hyperedges = incident_matrix.shape
    hyperedges_list = []

    for hyperedge_idx in range(num_hyperedges):
        nodes_in_hyperedge = np.where(incident_matrix[:, hyperedge_idx] == 1)[0].tolist()
        hyperedges_list.append(nodes_in_hyperedge)

    return hyperedges_list

H = ConstructH(data)
# print('-'*100)
# print(H.edge_index, H.edge_index.shape)
he_list = get_hyperedges_from_incident_matrix(H)
he_list = [tuple(i) for i in he_list]


data.edge_index = torch.LongTensor(data.edge_index)

#     Get splits
split_idx_lst = []
for run in range(args.runs):
    split_idx = rand_train_test_idx(
        data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
    split_idx_lst.append(split_idx)
    
# # Part 2: Load model


model = parse_method(args, data)

cvae_augmented_featuers, cvae_model = hgnn_cvae_pretrain_allset.get_augmented_features(args, data, he_list, model, split_idx, device)
torch.save(cvae_model,"model/%s_1231.pkl"%args.dataset)
# torch.save(cvae_augmented_featuers,"model/%s_augmented_features_1208.pkl"%args.dataset)

