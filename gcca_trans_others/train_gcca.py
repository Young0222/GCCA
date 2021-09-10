import argparse
import os.path as osp
import random
import pickle as pkl
import networkx as nx
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import sys
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor, Amazon, WikiCS
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification, LREvaluator


def sinkhorn(out, adj, D_inv, lambd, xi, iters=3):  # xi=0.05固定，lambd要小于xi
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

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def train(model: Model, x, edge_index, adj, D_inv, lambd, xi, nmb_prototypes):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    # 归一化
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    output_dim = z1.size(1)
    C = torch.nn.Linear(in_features=output_dim, out_features=nmb_prototypes, bias=False, device=z1.device)
    score_t = C(z1)
    score_s = C(z2)
    
    with torch.no_grad():
        q_t = sinkhorn(score_t, adj, D_inv, lambd, xi)
        q_s = sinkhorn(score_s, adj, D_inv, lambd, xi)

    temp = 0.1
    p_t = score_t / temp
    p_s = score_s / temp
    loss = - 0.5 * torch.mean(torch.sum(q_t * F.log_softmax(p_s, dim=0), dim=1) + torch.sum(q_s * F.log_softmax(p_t, dim=0), dim=1))
    loss.backward()
    optimizer.step()

    # normalize the prototypes
    with torch.no_grad():
        w = C.weight.data.clone()
        w = nn.functional.normalize(w, dim=0, p=2)
        C.weight.copy_(w)

    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def test_bgrl(encoder_model: Model, x, edge_index, y):
    encoder_model.eval()
    z = encoder_model(x, edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, y, split)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU(), })[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    nmb_prototypes = config['nmb_prototypes']
    lambd = config['lambd']
    xi = config['xi']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'CS', 'Physics', 'Computers', 'Photo', 'Wiki']
        name = 'dblp' if name == 'DBLP' else name
        if name in ['Cora', 'CiteSeer', 'PubMed']: 
            return Planetoid(
            path,
            name)
        elif name in ['CS', 'Physics']:
            return Coauthor(
            path,
            name,
            transform=T.NormalizeFeatures())
        elif name in ['Computers', 'Photo']:
            return Amazon(
            path,
            name,
            transform=T.NormalizeFeatures())
        elif name in ['Wiki']:
            return WikiCS(
            path,
            transform=T.NormalizeFeatures())
        else:
            return CitationFull(
            path,
            name)


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


    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    # adj = load_data(args.dataset.lower())
    data = dataset[0]
    edges = data.edge_index
    labels = data.y
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to_dense()
    N = adj.shape[0] # number of samples to assign
    K = nmb_prototypes # how many prototypes
    I_NN = torch.eye(N, device=adj.device)
    D =  I_NN - (lambd/xi) * (adj + adj.t())
    D_inv = D.inverse()
    # alpha = 0.2
    # N = adj.shape[0]
    # ppr = alpha * (torch.ones((N, N), device=adj.device) - (1 - alpha) * adj).inverse()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    adj = adj.to(device)
    D_inv = D_inv.to(device)
    # ppr = ppr.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index, adj, D_inv, lambd, xi, nmb_prototypes)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        # if epoch % 20 == 0:
        #     test_bgrl(model, data.x, data.edge_index, data.y)
        prev = now

    print("=== Final ===")
    for i in range(20):
        # test(model, data.x, data.edge_index, data.y, final=True)
        test_bgrl(model, data.x, data.edge_index, data.y)

