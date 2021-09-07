import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sinkhorn(self, out: torch.Tensor, adj: torch.Tensor, D_inv: torch.Tensor, lambd: float, 
                 xi: float, nmb_prototypes: int, iters: int = 3):  # xi=0.05固定，lambd要小于xi
        print("shape: ", out.shape, D_inv.shape)
        Q = torch.mm(out.t(), D_inv)
        Q = torch.exp(Q / xi)
        sum_Q = torch.sum(Q)    # make the matrix sums to 1
        Q /= sum_Q  # Q is K*N matrix
        N = adj.shape[0] # number of samples to assign
        K = nmb_prototypes # how many prototypes
        for it in range(iters):
            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            # normalize each column: total weight per sample must be 1/N
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= N
        Q *= N # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, C: torch.Tensor, adj: torch.Tensor, D_inv: torch.Tensor, 
                  lambd: float, xi: float, nmb_prototypes: int):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        score_t = C(z1)
        score_s = C(z2)
        
        with torch.no_grad():
            q_t = self.sinkhorn(score_t, adj, D_inv, lambd, xi, nmb_prototypes)

        temp = 0.1
        p_s = score_s / temp
        return torch.sum(q_t * F.log_softmax(p_s, dim=0), dim=1)
        

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, C: torch.Tensor, adj: torch.Tensor, D_inv: torch.Tensor, 
                          lambd: float, xi: float, nmb_prototypes: int, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes, device=z1.device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            loss = self.semi_loss(z1[mask], z2, C, adj, D_inv, lambd, xi, nmb_prototypes)
            losses.append(loss)

        return torch.cat(losses)
        

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, C: torch.Tensor, adj: torch.Tensor, D_inv: torch.Tensor, lambd: float, 
             xi: float, nmb_prototypes: int, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2, C, adj, D_inv, lambd, xi, nmb_prototypes)
            l2 = self.semi_loss(h2, h1, C, adj, D_inv, lambd, xi, nmb_prototypes)
        else:
            l1 = self.batched_semi_loss(h1, h2, C, adj, D_inv, lambd, xi, nmb_prototypes, batch_size)
            l2 = self.batched_semi_loss(h2, h1, C, adj, D_inv, lambd, xi, nmb_prototypes, batch_size)

        if mean:
            ret = - 0.5 * torch.mean(l1 + l2)
        else:
            ret = - 0.5 * torch.sum(l1 + l2)

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
