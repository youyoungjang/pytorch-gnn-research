# Graph Neural Networks with Pytorch
# Target: GraphSAGE
# Original Paper: https://arxiv.org/abs/1706.02216
# Original Code: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py

import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

sys.path.append('../')
from utils import *
logger = make_logger(name='graphsage_logger')


# Load Reddit Dataset
path = os.path.join(os.getcwd(), 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

# Data 확인
# Nodes: 232965, Node Features: 602
logger.info(f"Node Feature Matrix Info: # Nodes: {data.x.shape[0]}, # Node Features: {data.x.shape[1]}")

# Edge Index
# Graph Connectivity in COO format with shape (2, num_edges) = (2, 114,615,892=1.14억)
logger.info(f"Edge Index Shape: {data.edge_index.shape}")
logger.info(f"Edge Weight: {data.edge_attr}")

# train_mask denotes against which nodes to train (153431 nodes)
print(data.train_mask.sum().item())


# Define Sampler
train_loader = NeighborSampler(
    data.edge_index, node_idx=data.train_mask,
    sizes=[25, 10], batch_size=1024, shuffle=True, num_workers=12)

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=1024, shuffle=False, num_workers=12)

# Look
batch_size, n_id, adjs = next(iter(train_loader))

# 1) batch_size
# 현재 batch size를 의미함 (integer)
logger.info(f"Current Batch Size: {batch_size}")

# 2) n_id
# 이번 Subgraph에서 사용된 모든 node id
# batch_size개의 Row를 예측하기 위해서 이에 대한 1차 이웃 node A개가 필요하고
# 1차 이웃 node A개를 위해서는 2차 이웃 node B개가 필요함
# n_id.shape = batch_size + A + B
logger.info(f"현재 Subgraph에서 사용된 모든 node id의 개수: {n_id.shape[0]}")

# 3) adjs
# 아래와 같이 Layer의 수가 2개이면 adjs는 길이 2의 List가 된다.
# head node가 있고 1-hop neighbors와 2-hop neighbors가 있다고 할 때
# adjs[1]이 head node와 1-hop neighbors의 관계를 설명하며  (1번째 Layer)
# adjs[0]이 1-hop neighbors와 2-hop neighbors의 관계를 설명한다. (2번째 Layer)
logger.info(f"Layer의 수: {len(adjs)}")

# 각 리스트에는 아래와 같은 튜플이 들어있다.
# (edge_index, e_id, size)
# edge_index: source -> target nodes를 기록한 bipartite edges
# e_id: 위 edge_index에 들어있는 index가 Full Graph에서 갖는 node id

# size: 위 edge_index에 들어있는 node의 수를 튜플로 나타낸 것으로
# head -> 1-hop 관계를 예시로 들면,
# head node의 수가 a개, 1-hop node의 수가 b개라고 했을 때
# size = (a+b, a)
# 또한 target node의 경우 source nodes의 리스트의 시작 부분에 포함되어 있어
# skip-connections나 self-loops를 쉽게 사용할 수 있게 되어 있음
A = adjs[1].size[0] - batch_size
B = adjs[0].size[0] - A - batch_size

logger.info(f"진행 방향: {B}개의 2-hop neighbors ->"
            f"{A}개의 1-hop neighbors -> {batch_size}개의 Head Nodes")


# Define Model
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)

            # 마지막 Layer는 Dropout을 적용하지 않는다.
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())
    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()
    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
    return results


for epoch in range(1, 11):
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

