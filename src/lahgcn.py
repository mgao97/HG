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
import cvae_pretrain_hgcn

import time
from copy import deepcopy
# from config import config
import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import CocitationCora, Cooking200
# from data_load_utils import *
# from dhg.models import HGNN, LAHGCN
# from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from utils import accuracy, normalize_features, sparse_mx_to_torch_sparse_tensor, normalize_adj
from hgcn import *
from hgcn.models import HGNN, LAHGCN
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--concat", type=int, default=4)
parser.add_argument('--runs', type=int, default=5, help='The number of experiments.')
parser.add_argument("--latent_size", type=int, default=10)
# parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument("--pretrain_epochs", type=int, default=50)
parser.add_argument("--pretrain_lr", type=float, default=0.01)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
# parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

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
evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

# args = config.parse()



# seed
import os, torch, numpy as np
torch.manual_seed(args.seed)
np.random.seed(args.seed)



# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)



# load data
data = Cooking200()
print(data)

hg = Hypergraph(data["num_vertices"], data["edge_list"])
train_mask = data["train_mask"]
val_mask = data["val_mask"]
test_mask = data["test_mask"]

idx_train = np.where(train_mask)[0]
idx_val = np.where(val_mask)[0]
idx_test = np.where(test_mask)[0]

v_deg= hg.D_v
X = v_deg.to_dense()/torch.max(v_deg.to_dense())

# X = data["features"]

# Normalize adj and features
# features = data["features"].numpy()
features = X.numpy()
features_normalized = normalize_features(features)
# nor_hg = normalize_adj(hg)

# To PyTorch Tensor
# labels = torch.LongTensor(labels)
# labels = torch.max(labels, dim=1)[1]
labels = data["labels"]
features_normalized = torch.FloatTensor(features_normalized)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# cvae_model = torch.load("{}/model/{}.pkl".format(exc_path, args.dataset))

best_augmented_features, cvae_model = cvae_pretrain_hgcn.generated_generator(args, device, hg, X, labels, features_normalized, idx_train)

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = cvae_pretrain_hgcn.feature_tensor_normalize(augmented_features).detach()
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
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
for i in trange(args.runs, desc='Run Train'):

    # Model and optimizer
    model = LAHGCN(concat=args.concat+1,
                  in_channels=features.shape[1],
                  hid_channels=args.hidden,
                  num_classes=labels.max().item() + 1,
                  dropout=args.dropout)
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
        

        print('Epoch: {:04d}'.format(epoch+1),
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

    all_val.append(acc_val.item())
    all_test.append(acc_test.item())

print(np.mean(all_val), np.std(all_val), np.mean(all_test), np.std(all_test))