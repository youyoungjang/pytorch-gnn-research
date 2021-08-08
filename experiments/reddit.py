# Exploring Reddit Data
import os
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler


sys.path.append('../')
from utils import *
logger = make_logger(name='reddit_logger')


# Load Reddit Dataset
path = os.path.join(os.getcwd(), 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

# Data 확인
# Nodes: 232965, Node Features: 602, Edge Index: (2, 1.14억)
num_communities = len(set(data.y.numpy().tolist()))

logger.info(f"Node Feature Matrix Info: # Nodes: {data.x.shape[0]}")
logger.info(f"Node Feature Matrix Info: # Node Features: {data.x.shape[1]}")
logger.info(f"Edge Index Shape: {data.edge_index.shape}")
logger.info(f"Edge Weight: {data.edge_attr}")
logger.info(f"# Community: {num_communities}")

# train_mask denotes against which nodes to train (153431 nodes)
print(data.train_mask.sum().item())

# torch_geometric datasets에 있는 Reddit 데이터는 GraphSAGE 논문에 있는 설명을 따른다.
# 이 데이터에서 node는 post이고, node feature는 post의 내용을 임베딩한 것이다.
# 같은 User가 Comment를 남긴 post 사이에는 link가 있다고 설정하며
# 이러한 post들은 어떤 Community에 속하게 된다.
# 총 41개의 Community가 데이터셋에 포함되어 있으며 이는 위에서 본 것처럼 data.y에서 확인할 수 있다.
# 예측 Task는 각 Node(Post)가 어떤 Community에 속하는지 파악하는 것이다.

# Link의 분포
# 최소: 1, 최대: 21657 (엄청나다)
source_nodes = data.edge_index[0, :]
report = torch.bincount(input=source_nodes, weights=None, minlength=0)

max_link_node = torch.argmax(report)
max_links = report[max_link_node]

min_link_node = torch.argmin(report)
min_links = report[min_link_node]

# indices = torch.where(data.edge_index[0, :] == max_link_node)[0]
logger.info(f"가장 많은 link를 갖는 node: {max_link_node} with {max_links} links")
logger.info(f"가장 적은 link를 갖는 node: {min_link_node} with {min_links} links")


# check torch_geomectic.data.Data
print(data.contains_isolated_nodes())
print(data.contains_self_loops())
print(data.is_directed())



# Create Data Loader
num_samples = [10, 5]
num_layers = len(num_samples)

train_neigh_sampler = NeighborSampler(
    data.edge_index, node_idx=data.train_mask+data.val_mask,
    sizes=num_samples, batch_size=1024, shuffle=True, num_workers=0)

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=1024, shuffle=False, num_workers=0)


# Check
batch_size, n_id, adjs = next(iter(train_neigh_sampler))

logger.info(f"Current Batch Size: {batch_size}")
logger.info(f"현재 Subgraph에서 사용된 모든 node id의 개수: {n_id.shape[0]}")
logger.info(f"Layer의 수: {len(adjs)}")

A = adjs[1].size[0] - batch_size
B = adjs[0].size[0] - A - batch_size

logger.info(f"진행 방향: {B}개의 2-hop neighbors ->"
            f"{A}개의 1-hop neighbors -> {batch_size}개의 Head Nodes")


# Define Model
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, drop_rate, num_node_features, num_layers, hidden_size, out_channels):
        super(GIN, self).__init__()
        self.drop_rate = drop_rate
        self.num_node_features = num_node_features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_channels = out_channels

        self.gin_convs = torch.nn.ModuleList()
        self.gin_convs.append(
            GINConv(
                nn=Sequential(
                    Linear(in_features=num_node_features, out_features=hidden_size),
                    BatchNorm1d(num_features=hidden_size),
                    ReLU(),
                    Dropout(p=drop_rate),
                    Linear(hidden_size, hidden_size),
                    ReLU()
                )
            )
        )
        self.gin_convs.append(
            GINConv(
                nn=Sequential(
                    Linear(in_features=hidden_size, out_features=hidden_size),
                    BatchNorm1d(num_features=hidden_size),
                    ReLU(),
                    Dropout(p=drop_rate),
                    Linear(hidden_size, out_channels)
                )
            )
        )

    def forward(self, x_batch, adjs):
        # x_batch = X[n_id] or data.x[n_id]
        # x_batch = head_node_features + 1_hop_node_features + 2_hop_node_features
        for i, (edge_index, e_id, size) in enumerate(adjs):
            # size ex) (10399, 1024) or (45887, 10399)
            x_target = x_batch[:size[1]]
            x_batch = self.gin_convs[i]((x_batch, x_target), edge_index)

        out = F.log_softmax(x_batch, dim=-1)
        return out

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Testing')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.gin_convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(
    drop_rate=0.2, num_node_features=data.num_node_features, num_layers=num_layers,
    hidden_size=128, out_channels=num_communities
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X = data.x.to(device)
Y = data.y.squeeze().to(device)

#batch_size, n_id, adjs = next(iter(train_neigh_sampler))
#adjs = [adj.to(device) for adj in adjs]
#x_batch = X[n_id]
#out = model(x_batch, adjs)

# Train & Test functions
def train(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_neigh_sampler:
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(X[n_id], adjs)
        loss = F.nll_loss(out, Y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(Y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_neigh_sampler)
    approx_acc = total_correct / int(data.train_mask.sum()+data.val_mask.sum())
    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()
    out = model.inference(X)

    y_true = Y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask+data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
    return results


# Main
for epoch in range(1, 21):
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

    train_acc, test_acc = test()
    print(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')


# Epoch 20, Loss: 0.3046, Train Acc: 0.8916, Test ACC: 0.8949
