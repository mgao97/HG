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
import hgnn_cvae_pretrain_new_news20

import time
from copy import deepcopy
# from config import config
import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import *
# from data_load_utils import *
# from dhg.models import HGNN, LAHGCN
# from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from utils import accuracy, normalize_features, micro_f1, macro_f1, sparse_mx_to_torch_sparse_tensor, normalize_adj
from hgcn import *
from hgcn.models import LAUniSAGE
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
exc_path = sys.path[0]
import os, torch, numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


from convert_datasets_to_pygDataset import dataset_Hypergraph
from layers import *
from models import *
from preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=10)
parser.add_argument("--concat", type=int, default=4)
parser.add_argument('--runs', type=int, default=1, help='The number of experiments.')

parser.add_argument("--pretrain_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=0.01)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
# parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

# parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument('--dataset', default='news20', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')

# parser.add_argument("--pretrain_epochs", type=int, default=8)
# parser.add_argument("--pretrain_lr", type=float, default=0.01)
# parser.add_argument("--conditional", action='store_true', default=True)
# parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
# parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
# parser.add_argument('--warmup', type=int, default=200, help='Warmup')
# parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')



parser.add_argument('--dname', default='20newsW100')
parser.add_argument('--add_self_loop', action='store_false')
parser.add_argument('--exclude_self', action='store_true')
parser.add_argument('--normtype', default='all_one')

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
# Load data
existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                    'NTU2012', 'Mushroom',
                    'coauthor_cora', 'coauthor_dblp',
                    'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                    'walmart-trips-100', 'house-committees-100',
                    'cora', 'citeseer', 'pubmed']

if args.dname in existing_dataset:
    dname = args.dname
    # f_noise = args.feature_noise
    p2raw = '../data/AllSet_all_raw_data/'
    dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
    data = dataset.data
    data = ExtractV2E(data)
    # if args.add_self_loop:
    #         data = Add_Self_Loops(data)
    if args.exclude_self:
        data = expand_edge_index(data)

    data = norm_contruction(data, option=args.normtype)


print('data:',data)

H = np.load('H.npy')
print(H.shape)
print(H)


# 初始化超边列表
he = []

for j in range(len(H[0])):  # 遍历每一列
    edge_vertices = set()
    for i in range(len(H)):  # 遍历每一行
        if H[i][j] != 0:
            edge_vertices.add(i)  # 添加非零元素的行索引到集合中
    he.append(list(edge_vertices))  # 将集合转换为列表，并添加到超边列表中


# # 输出超边列表

# print(data.n_x,type(data.n_x))

hg = Hypergraph(int(data.n_x), he)
print(hg)

X = data.x
data['num_vertices'] = data.n_x


print(hg)
hg = hg.to(device)
# Normalize adj and features
features = X.numpy()
features_normalized = normalize_features(features)
labels = data.y
features_normalized = torch.FloatTensor(features_normalized).to(device)
num_vertices = int(data.n_x)
# data = News20()
# hg = Hypergraph(data["num_vertices"], data["edge_list"])
# print(hg)

# num_vertices = data['num_vertices']
# labels = data['labels']
# 设置随机种子，以确保结果可复现
random_seed = 42

node_idx = [i for i in range(num_vertices)]
# 将idx_test划分为训练（60%）、验证（20%）和测试（20%）集
idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# 确保划分后的集合没有重叠
assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0



# v_deg= hg.D_v
# X = v_deg.to_dense()/torch.max(v_deg.to_dense())

# # X = data["features"]

# # Normalize adj and features
# # features = data["features"].numpy()
# features = X.numpy()
# features_normalized = normalize_features(features)

# features_normalized = torch.FloatTensor(features_normalized)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


train_mask = torch.zeros(num_vertices, dtype=torch.bool)
val_mask = torch.zeros(num_vertices, dtype=torch.bool)
test_mask = torch.zeros(num_vertices, dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

cvae_model = torch.load("{}/model/{}_0317.pkl".format(exc_path, args.dataset))
cvae_model = cvae_model.to(device)
# best_augmented_features, cvae_model = hgnn_cvae_pretrain_new_cora.get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)

# def get_augmented_features(concat):
#     X_list = []
#     cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
#     for _ in range(concat):
#         z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
#         augmented_features = cvae_model.inference(z, cvae_features)
#         augmented_features = hgnn_cvae_pretrain_new_news20.feature_tensor_normalize(augmented_features).detach()
#         if args.cuda:
#             X_list.append(augmented_features.to(device))
#         else:
#             X_list.append(augmented_features)
#     return X_list

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
        
        batch_size = 16
        from torch.utils.data import DataLoader, TensorDataset

        # 将数据和标签组成一个 TensorDataset
        cvae_data = TensorDataset(z, cvae_features)
        cvae_features_loader = DataLoader(cvae_data, batch_size=batch_size, shuffle=True)#, num_workers=8)
        batch_res = []
        for batch_z, cvae_features_batch in cvae_features_loader:
            
            z_batch, cvae_features_batch = batch_z.to(device), cvae_features_batch.to(device)
            # print('z_batch.shape:',z_batch.shape, 'cvae_features_batch.shape:',cvae_features_batch.shape)
            augmented_features = cvae_model.inference(z_batch, cvae_features_batch)
            # print('augmented_features:', augmented_features.shape)
            augmented_features = hgnn_cvae_pretrain_new_news20.feature_tensor_normalize(augmented_features).detach()
            if args.cuda:
                batch_res.append(augmented_features.to(device))
            else:
                batch_res.append(augmented_features)
            
            
            batch_list = torch.cat(batch_res, dim=0)
        # print('len(batch_res)', len(batch_res))
        # print('len(batch_res[0])', batch_res[0].shape)
        # print('batch list shape:', batch_list.shape)
        X_list.append(batch_list)
        del batch_list
        torch.cuda.empty_cache()  # 释放 GPU 存储
        print('done!')
        
    return X_list


if args.cuda:
    hg = hg.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

all_val = []
all_test = []

all_test_microf1, all_test_macrof1 = [], []


for i in trange(args.runs, desc='Run Train'):

    # Model and optimizer
    model = LAUniSAGE(concat=args.concat+1,
                  in_channels=features.shape[1],
                  hid_channels=args.hidden,
                  num_classes=labels.max().item() + 1,
                  use_bn=False,
                  drop_rate=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            output_list.append(torch.log_softmax(model(X_list+[features_normalized], hg), dim=-1))

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
        output = model(val_X_list+[features_normalized],hg)
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

    #Validate and Test
    best_model.eval()
    output = best_model(best_X_list+[features_normalized], hg)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # micro_f1_val = micro_f1(output[idx_val], labels[idx_val])
    # macro_f1_val = macro_f1(output[idx_val], labels[idx_val])
    micro_f1_test = micro_f1(output[idx_test], labels[idx_test])
    macro_f1_test = macro_f1(output[idx_test], labels[idx_test])

    all_val.append(acc_val.item())
    all_test.append(acc_test.item())
    all_test_microf1.append(micro_f1_test.item())
    all_test_macrof1.append(macro_f1_test.item())

# print('val acc:', np.mean(all_val), 'val acc std:', np.std(all_val))
# print('\n')
print('test acc:', np.mean(all_test), 'test acc std', np.std(all_test))
print('\n')
print('test micro f1:', np.mean(all_test_microf1), 'test macro f1', np.mean(all_test_macrof1))