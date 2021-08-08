# Graph Neural Networks with Pytorch
# Target: Graph Convolutional Networks
# Original Paper: https://arxiv.org/abs/1609.02907
# Original Code: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py

import sys
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

sys.path.append('../')
from utils import *
logger = make_logger(name='gcn_logger')

# Load Cora Dataset
dataset = 'Cora'
path = osp.join(os.getcwd(), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

# Data Check
# 2708 nodes exist, node feature length is 1433
logger.info(f"Node Feature Matrix Info: # Nodes: {data.x.shape[0]}, # Node Features: {data.x.shape[1]}")
logger.info(f"Target Y Info: Y={data.y.size()}")

# Edge Index
# Graph Connectivity in COO format with shape (2, num_edges) = (2, 10556)
# ([[   0,    0,    0,  ..., 2707, 2707, 2707],
#   [ 633, 1862, 2582,  ...,  598, 1473, 2706]])
logger.info(f"Edge Index Shape: {data.edge_index.shape}")
logger.info(f"Edge Weight: {data.edge_attr}")

# So graph is unweighted and undirected
# Custom Dataset can be made with torch_geometric.data.Data

# components of data can be optained by dictionary format
print(data.keys)

# Other Attributes
# num_nodes, num_edges, contains_isolated_nodes(), contains_self_loops(), is_directed()

# train_mask denotes against which nodes to train (140 nodes)
print(data.train_mask.sum().item())

# Define Model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(
            in_channels=dataset.num_features, out_channels=16, cached=True, normalize=True)
        self.conv2 = GCNConv(
            in_channels=16, out_channels=dataset.num_classes, cached=True, normalize=True)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mark that you should also transfer data object to GPU
model, data = GCN().to(device), data.to(device)

# Only perform weight-decay on first convolution.
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(
        input=model()[data.train_mask], target=data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
