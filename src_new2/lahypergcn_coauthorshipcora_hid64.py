from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
import hgnn_cvae_pretrain_new_house
from dhg.structure.graphs import Graph
from dhg.nn import MultiHeadWrapper
import time
from copy import deepcopy
# from config import config
import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import CoauthorshipCora
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from utils import accuracy, normalize_features, micro_f1, macro_f1, sparse_mx_to_torch_sparse_tensor, normalize_adj
from hgcn import *
from hgcn.models import LAHyperGCN
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
exc_path = sys.path[0]
import os, torch, numpy as np
import hgnn_cvae_pretrain_new_coauthorcora
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--concat", type=int, default=8)
parser.add_argument('--runs', type=int, default=1, help='The number of experiments.')

parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument('--dataset', default='coauthorcora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument("--pretrain_epochs", type=int, default=8)
parser.add_argument("--pretrain_lr", type=float, default=0.05)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')

parser.add_argument('--use_mediator', default=False, help='Whether to use mediator to transform the hyperedges to edges in the graph.')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

print('args\n', args)

# Adapted from GRAND: https://github.com/THUDM/GRAND
def consis_loss(logps, temp=args.tem):
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


# Load data
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)



# load data
data = CoauthorshipCora()
print(data)

G = Hypergraph(data["num_vertices"], data["edge_list"])

# he = G.e[0]
# G.e[0][27] = (447, 0)
labels = data["labels"]

# # 设置随机种子，以确保结果可复现
random_seed = 42

node_idx = [i for i in range(data["num_vertices"])]
labels = data["labels"]
y = labels.numpy()
# 将idx_test划分为训练（60%）、验证（20%）和测试（20%）集
idx_train, idx_temp, train_y, tem_y = train_test_split(node_idx, y, test_size=0.5, random_state=random_seed, stratify=y)
idx_val, idx_test, val_y, test_y = train_test_split(idx_temp, tem_y, test_size=0.5, random_state=random_seed, stratify=tem_y)
# # 将idx_test划分为训练（50%）、验证（25%）和测试（25%）集
# idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
# idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# 确保划分后的集合没有重叠
assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0


train_mask = torch.zeros(G.num_v, dtype=torch.bool)
val_mask = torch.zeros(G.num_v, dtype=torch.bool)
test_mask = torch.zeros(G.num_v, dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True


X = data['features']

num_epochs = 200

X, G = X.to(device), G.to(device)
features = X.numpy()
features_normalized = normalize_features(features)
features_normalized = torch.FloatTensor(features_normalized)

cvae_model = torch.load("{}/model/{}_0104.pkl".format(exc_path, args.dataset))

# best_augmented_features, cvae_model = hgnn_cvae_pretrain_new_cora.get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(X, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
        # print('z.shape, cvae_features.shape:',z.shape, cvae_features.shape)
        augmented_features = cvae_model.inference(z, cvae_features)
        # print('augmented_features', augmented_features.shape)
        augmented_features = hgnn_cvae_pretrain_new_coauthorcora.feature_tensor_normalize(augmented_features).detach()
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list


if args.cuda:
    G = G.to(device)
    labels = torch.cat((labels, torch.LongTensor([2])), dim=0).to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

all_val = []
all_test = []

all_test_microf1, all_test_macrof1 = [], []


for i in trange(args.runs, desc='Run Train'):

    # Model and optimizer
    model = LAHyperGCN(concat=args.concat+1,
                  in_channels=X.shape[1],
                  hid_channels=args.hidden,
                  num_classes=labels.max().item() + 1,
                  use_mediator=args.use_mediator,
                  use_bn=False,
                  drop_rate=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('model:',model)
    if args.cuda:
        model.to(device)

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
            output_list.append(torch.log_softmax(model(X_list+[features_normalized], G), dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
        loss_train = loss_train/len(output_list)

        loss_consis = consis_loss(output_list)
        loss_train = loss_train + loss_consis

        loss_train.backward()
        optimizer.step()

        model.eval()
        val_X_list = get_augmented_features(args.concat)
        output = model(val_X_list+[features_normalized], G)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        

        print('Run:{:02d}'.format(i+1),
              'Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()))
                
        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            best_X_list = copy.deepcopy(val_X_list)
            torch.save(best_model.state_dict(), 'model/lahypergcn_coauthorcora_hid64_0107.pth')


    #Validate and Test
    best_model.eval()
    output = best_model(best_X_list+[features_normalized], G)
    output = torch.log_softmax(output, dim=1)
    
    # print('='*100)
    # print('output:', output.shape, 'lbls:',labels.shape)
    # print('='*100)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    micro_f1_test = micro_f1(output[idx_test], labels[idx_test])
    macro_f1_test = macro_f1(output[idx_test], labels[idx_test])

    all_val.append(acc_val.item())
    all_test.append(acc_test.item())
    all_test_microf1.append(micro_f1_test.item())
    all_test_macrof1.append(macro_f1_test.item())

print('test acc:', np.mean(all_test), 'test acc std', np.std(all_test))
print('\n')
print('test micro f1:', np.mean(all_test_microf1), 'test macro f1', np.mean(all_test_macrof1))