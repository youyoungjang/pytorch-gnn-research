# Graph Neural Networks with Pytorch
# Target: APPNP: Predict Then Propagate
# Original Paper: https://arxiv.org/abs/1810.05997
# Original Code: https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/appnp.py

import os
import sys
import time
from typing import KeysView
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



# Define Model
class Net(torch.nn.Module):
    def __init__(self, dataset, K, alpha):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, 64)
        self.lin2 = Linear(64, dataset.num_classes)
        self.prop1 = APPNP(K, alpha)

        self.K = K
        self.alpha = alpha

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


data = dataset[0]
data = permute_masks(data, dataset.num_classes)
data = data.to(device)


# Hyper-parameter Tuning with Ray Tune
from functools import partial

# pip install ray[tune]
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


# Configure the search space
config = {
    "weight_decay": tune.sample_from(lambda _: 5/(10**np.random.randint(4, 6))),
    "lr": tune.loguniform(lower=1e-4, upper=1e-1),
    "K": tune.choice([10, 20, 30, 40, 50]),
}


def proceed(config, checkpoint_dir=None):
    model = Net(
        dataset=dataset,
        K=config['K'],
        alpha=0.1
    )

    # Multiple GPU check
    if torch.cuda.device_count():
        model = torch.nn.DataParallel(model)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint'))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    epochs = 10
    for epoch in range(1, epochs+1):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()

        with torch.no_grad():
            logits = model(data)

        # eval_info: dictionary containing loss, acc
        eval_info = {}
        for key in ['train', 'val']:
            mask = data['{}_mask'.format(key)]
            loss = F.nll_loss(logits[mask], data.y[mask]).item()
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            eval_info['{}_loss'.format(key)] = loss
            eval_info['{}_acc'.format(key)] = acc

        # Ray Tune
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=eval_info['val_loss'], accuracy=eval_info['val_acc'])


# main
num_samples = 10
max_num_epochs = 10


scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2)

reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"])

checkpoint_dir = os.path.join(os.getcwd(), 'appnp/checkpoint')


result = tune.run(
    partial(proceed, checkpoint_dir=checkpoint_dir),
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter)


best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))


best_trained_model = Net(dataset=dataset, K=best_trial.config['K'])


