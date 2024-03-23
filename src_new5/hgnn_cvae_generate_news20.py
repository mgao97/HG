import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import hgnn_cvae_pretrain_new_news20
import faulthandler
faulthandler.enable()
from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
# from gcn.models import GCN
from hgnn_cvae_pretrain import HGNN
from tqdm import trange
import dhg
from dhg.data import *
from dhg import Hypergraph
from dhg.nn import HGNNConv
from sklearn.model_selection import train_test_split
from convert_datasets_to_pygDataset import dataset_Hypergraph
from layers import *
from models import *
from preprocessing import *

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=0.1)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

parser.add_argument('--dataset', default='news20',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

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

print('args:\n', args)



def bipartite_representation_from_adjacency_matrix(A, N):
    """
    将二分表示的邻接矩阵 A 转换为关联矩阵 C。

    参数：
    - A: 二分表示的邻接矩阵，大小为 (N+M) x (N+M)。
    - N: 节点数。

    返回：
    - C: 关联矩阵，大小为 N x M。
    """

    M = A.shape[1] - N  # 超边数

    # 初始化关联矩阵 C
    C = np.zeros((N, M))

    # 将 A 中的元素映射到 C 的对应位置
    for i in range(N):
        for j in range(N, N+M):
            C[i, j-N] = A[i, j]

    return C

import torch

def build_adjacency_matrix(edge_index):
    
    num_edges = edge_index.shape[1]
    
    # 计算总顶点数
    total_vertices = edge_index.max().item() + 1
    
    # 初始化邻接矩阵，一开始都是0
    adjacency_matrix = torch.zeros((total_vertices, total_vertices), dtype=torch.int8)
    
    # 遍历所有边，设置邻接矩阵中对应的元素为1
    for i in range(num_edges):
        # 获取边的两个顶点
        edge = edge_index[:, i]
        
        # 设置邻接矩阵中对应的元素为1
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1  # 由于二分图是对称的，也需要设置另一个方向的边
    
    return adjacency_matrix





# Load data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Cocitation Cora Data
### Load and preprocess data ###
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


# 二分图邻接矩阵转超图关联矩阵
# A = build_adjacency_matrix(data.edge_index)

# # 打印邻接矩阵
# print(A,A.shape)

# H = bipartite_representation_from_adjacency_matrix(A, N=data.n_x)

# print(H,H.shape)
# np.save('H.npy', H)

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
print(X,X.shape)


data['num_vertices'] = data.n_x

# # # for datasets without initial feats
# v_deg= hg.D_v
# # data["features"] = v_deg.to_dense()/torch.max(v_deg.to_dense())
# X = v_deg.to_dense()/torch.max(v_deg.to_dense())

# Normalize adj and features
features = X.numpy()
features_normalized = normalize_features(features)
labels = data.y
features_normalized = torch.FloatTensor(features_normalized)

# 设置随机种子，以确保结果可复现
random_seed = 42

node_idx = [i for i in range(data['num_vertices'])]
# 将idx_test划分为训练（60%）、验证（20%）和测试（20%）集
idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

# 确保划分后的集合没有重叠
assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


train_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
val_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
test_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

cvae_augmented_featuers, cvae_model = hgnn_cvae_pretrain_new_news20.get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)
torch.save(cvae_model,"model/%s_0317.pkl"%args.dataset)

# torch.save(cvae_augmented_featuers,"model/%s_augmented_features_1208.pkl"%args.dataset)

