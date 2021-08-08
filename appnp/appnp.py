# Graph Neural Networks with Pytorch
# Target: APPNP: Predict Then Propagate
# Original Paper: https://arxiv.org/abs/1810.05997
# Original Code: https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/appnp.py

import os
import sys
import time
import numpy as np

import torch
from torch import tensor
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import APPNP
from torch.optim import Adam
from torch_geometric.datasets import Planetoid


sys.path.append('../')
from utils import *
logger = make_logger(name='appnp_logger')


# Load Data
path = os.path.join(os.getcwd(), 'data', 'Cora')
dataset = Planetoid(path, 'Cora')
dataset.transform = T.NormalizeFeatures()


# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# permute_mask = random_planetoid_splits 함수
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
    return data

permute_masks = random_planetoid_splits


# Permute Masks
# 원래 데이터는 train_mask, valid_mask, test_mask가 순서대로 정렬되어 있음
data = dataset[0]
print(data.train_mask[0:20])

# 랜덤하게재 배치됨
data = permute_masks(data, dataset.num_classes)
print(data.train_mask[0:20])

data = data.to(device)


# Environments
HIDDEN_SIZE = 64
ALPHA = 0.1
DROP_RATE = 0.5


# Define Model
class Net(torch.nn.Module):
    def __init__(self, dataset, K):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, HIDDEN_SIZE)
        self.lin2 = Linear(HIDDEN_SIZE, dataset.num_classes)
        self.prop1 = APPNP(K, ALPHA)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=DROP_RATE, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=DROP_RATE, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, data):
    # Train mode
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    # Test mode
    model.eval()

    with torch.no_grad():
        logits = model(data)

    # dictionary containing loss, acc
    eval_info = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        eval_info['{}_loss'.format(key)] = loss
        eval_info['{}_acc'.format(key)] = acc
    return eval_info


# Model, Optimizer, Early Stopping
# 1) Model
model = Net(dataset, K=10).to(device)
model.reset_parameters()

message = {k:list(v.shape) for k, v in model.named_parameters()}
logger.info(f"Trainable Parameters: \n {message}")
# There is no parameter in APPNP Layer


# 2) Optimizer
lr = 0.01
weight_decay = 0.0005
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# 3) Early Stopping (utils.py 참조)
patience = 10
save_path = os.path.join(os.getcwd(), 'appnp/checkpoints/checkpoint.pnt')
early_stopping = EarlyStopping(patience=patience, save_path=save_path)

logger.info(f"Early Stopping Object created with patience {patience}")


# Pytorch에서 cuda 호출은 비동기식이기 때문에 시간을 재기 이전에 동기화가 필요함
# 참고: https://discuss.pytorch.org/t/best-way-to-measure-timing/39496
def run_model(model, optimizer, early_stopping, epochs=100):
    durations = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    best_val_loss = float('inf')
    val_loss_history = []

    # for run in range(runs):
    for epoch in range(1, epochs + 1):
        train(model, optimizer, data)
        eval_info = evaluate(model, data)
        eval_info['epoch'] = epoch

        # 최고의 버전을 저장함
        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            test_acc = eval_info['test_acc']
        val_loss_history.append(eval_info['val_loss'])

        print(
            f"Epoch: {epoch}, Train ACC: {eval_info['train_acc']:4f}, Val ACC: {eval_info['val_acc']:4f}, Test ACC: {eval_info['test_acc']:4f}")

        early_stopping(eval_info['val_loss'], model)

        # 최소 절반 정도 진행되었고 early_stop 조건이 만족하면
        if early_stopping.early_stop == True and epoch > epochs // 2:
            print("Early Stoping...")
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    durations = t_end - t_start

    print(f"took {durations:4f} seconds")

run_model(model, optimizer, early_stopping, epochs=100)


