import functools

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np
from sklearn.preprocessing import normalize


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            info = results[0][1]
            results = [r[0] for r in results]

            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, info, f.__name__)
            return statistics

        return wrapper

    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, info, function_name):
    print(f'({info}) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}~{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(1)
def label_classification_inductive(train_x, train_y, test_x, test_y, info):
    x_train = train_x.detach().cpu().numpy()
    y_train = train_y.detach().cpu().numpy()
    x_test = test_x.detach().cpu().numpy()
    y_test = test_y.detach().cpu().numpy()

    x_train = normalize(x_train, norm='l2')
    x_test = normalize(x_test, norm='l2')
    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    clf = LogisticRegression(n_jobs=16, solver='sag', max_iter=100)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    micro = f1_score(y_test, y_pred, average="micro")

    return {'F1Mi': micro,}, info


from torch_geometric.nn import SAGEConv

import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

from functools import wraps
import copy

# torch.manual_seed(0)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Encoder(torch.nn.Module):
    def __init__(self, layer_config):
        super(Encoder, self).__init__()
        self.act = nn.PReLU()
        dim = 512

        self.W = nn.Linear(layer_config[0], dim)
        self.W2 = nn.Linear(layer_config[0], dim)

        self.layer1 = SAGEConv(layer_config[0], dim)
        self.layer2 = SAGEConv(dim, dim)
        self.layer3 = SAGEConv(dim, 512)

    def forward(self, x, edge_index, edge_weight=None):
        h_1 = self.layer1(x, edge_index)
        h_1 = self.act(h_1)
        h_2 = self.layer2(h_1 + self.W(x), edge_index)
        h_2 = self.act(h_2)
        h_3 = self.layer3(h_2 + h_1 + self.W2(x), edge_index)
        h_3 = self.act(h_3)
        return h_3


class GCCA(nn.Module):

    def __init__(self, layer_config, dropout=0.0, moving_average_decay=0.99, epochs=1000, **kwargs):
        super().__init__()
        self.encoder = Encoder(layer_config=layer_config)

    def embed(self, x, edge_index):
        return self.encoder(x, edge_index).detach()

    def forward(self, x1, x2, edge_index_v1, edge_index_v2):
        return self.encoder(x1, edge_index_v1), self.encoder(x2, edge_index_v2)


import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI
from torch_geometric.utils import dropout_adj
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv
from tqdm import trange
import scipy.sparse as sp

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda is True:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def get_dataset(path, name, split):
    assert name in ['PPI']
    if name == 'PPI':
        return PPI(path, split, transform=T.NormalizeFeatures())


def sinkhorn(out, adj, D_inv, lambd, xi, K, N, iters=3):  # xi=0.05固定，lambd要小于xi
    Q = torch.mm(out.t(), D_inv)
    Q = torch.exp(Q / xi)
    sum_Q = torch.sum(Q)    # make the matrix sums to 1
    Q /= sum_Q  # Q is K*N matrix
    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        # normalize each column: total weight per sample must be 1/N
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= N
    Q *= N # the colomns must sum to 1 so that Q is an assignment
    return Q.t()
    

def train(model, data_loader, device):
    model.train()

    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # r_inv = np.power(rowsum, -1).flatten()
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv)
        return mx

    total_loss = 0
    for data in data_loader:
        # data.to(device)
        x, edge_index, labels = data.x, data.edge_index, data.y

        optimizer.zero_grad()

        edges = edge_index
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj = adj.to_dense()
        N = adj.shape[0] # number of samples to assign
        I_NN = torch.eye(N, device=adj.device)
        lambd = 0.005
        xi = 0.05
        D =  I_NN - (lambd/xi) * (adj + adj.t())
        D_inv = D.inverse()

        data = data.to(device)
        adj = adj.to(device)
        D_inv = D_inv.to(device)

        edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
        x_1 = drop_feature(x, drop_feature_rate_1)
        x_2 = drop_feature(x, drop_feature_rate_2)

        x_1 = x_1.to(device)
        x_2 = x_2.to(device)
        edge_index_1 = edge_index_1.to(device)
        edge_index_2 = edge_index_2.to(device)
        z1, z2 = model(x_1, x_2, edge_index_1, edge_index_2)
        # 归一化
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        output_dim = z1.size(1)
        nmb_prototypes = 100
        C = torch.nn.Linear(in_features=output_dim, out_features=nmb_prototypes, bias=False, device=z1.device)
        score_t = C(z1)
        score_s = C(z2)
        
        with torch.no_grad():
            q_t = sinkhorn(score_t, adj, D_inv, lambd, xi, nmb_prototypes, N)
            q_s = sinkhorn(score_s, adj, D_inv, lambd, xi, nmb_prototypes, N)

        temp = 0.1
        p_t = score_t / temp
        p_s = score_s / temp
        loss = - 0.5 * torch.mean(torch.sum(q_t * F.log_softmax(p_s, dim=0), dim=1) + torch.sum(q_s * F.log_softmax(p_t, dim=0), dim=1))

        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()

        # normalize the prototypes
        with torch.no_grad():
            w = C.weight.data.clone()
            w = nn.functional.normalize(w, dim=0, p=2)
            C.weight.copy_(w)

    return total_loss / len(train_loader.dataset)


def eval(model, dataset_train, dataset_test, device=None):
    model.eval()

    train_emb, train_label = [], []
    for d in dataset_train:
        d.to(device)
        h = model.embed(d.x, d.edge_index)
        train_emb.append(h)
        train_label.append(d.y)
    train_emb = torch.cat(train_emb, dim=0)
    train_label = torch.cat(train_label, dim=0)
    print("train is finished!")

    test_emb, test_label = [], []
    for d in dataset_test:
        d.to(device)
        h = model.embed(d.x, d.edge_index)
        test_emb.append(h)
        test_label.append(d.y)
    test_emb = torch.cat(test_emb, dim=0)
    test_label = torch.cat(test_label, dim=0)
    print("test is finished!")


    for i in range(20):
        label_classification_inductive(train_emb, train_label, test_emb, test_label, info='GCCA.')


if __name__ == '__main__':
    dataset = 'PPI'
    activation = nn.PReLU()
    base_model = SAGEConv
    num_hidden = 512

    drop_edge_rate_1 = 0.3
    drop_edge_rate_2 = 0.25
    drop_feature_rate_1 = .25
    drop_feature_rate_2 = .0

    num_epochs = 100
    learning_rate = 0.0001
    weight_decay = 0.00001

    print(
        f"GCCA | dataset:{dataset} | epochs:{num_epochs} | "
        f"drop_rate:{drop_edge_rate_1}-{drop_edge_rate_2}-{drop_feature_rate_1}-{drop_feature_rate_2}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join('dataset/PPI')
    dataset_train = get_dataset(path, dataset, 'train')
    dataset_val = get_dataset(path, dataset, 'val')
    dataset_test = get_dataset(path, dataset, 'test')
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=2, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=2, shuffle=False)

    model = GCCA(layer_config=[dataset_train[0].x.shape[1], num_hidden], epochs=num_epochs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_train = 1e9
    for epoch in trange(1, num_epochs + 1):
        # train
        loss = train(model, train_loader, device)
        if loss < best_train:
            best_train = loss
            torch.save(model.state_dict(), f'checkpoint/best_{dataset}_GCCA.pkl')

    model.load_state_dict(torch.load(f'checkpoint/best_{dataset}_GCCA.pkl'))
    print('testing...')
    eval(model, train_loader, test_loader, device)

print("All over!\n")

