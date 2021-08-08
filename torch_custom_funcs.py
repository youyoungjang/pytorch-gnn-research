# Pytorch Utility Functions

import os, sys
import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_sparse import masked_select_nnz


# convenience
class GraphInspector(object):
    def __init__(self, data):
        self.data = data

    def get_basic_info(self):
        info = {}
        if hasattr(self.data, 'num_nodes'):
            info['num_nodes'] = self.data.num_nodes
        if hasattr(self.data, 'num_edges'):
            info['num_edges'] = self.data.num_edges
        print(info)

    def inspect(self, attribute: str):
        data = self.data
        if attribute == 'edge_index' and hasattr(data, 'edge_index'):
            most_freq_appeared_node = torch.argmax(torch.bincount(data.edge_index[0, :]))
            cnt = torch.bincount(data.edge_index[0, :])[most_freq_appeared_node]

            print(f"most_freq_appeared_node: {most_freq_appeared_node} with {cnt}")
        else:
            ValueError("data has no attribute named edge_index")

        if attribute == 'edge_type' and hasattr(data, 'edge_type'):
            num_edge_type = torch.unique(data.edge_type).size(0)
            cnt_per_edge_type = torch.bincount(data.edge_type)

            most_freq_appeared_edge_type = torch.argmax(torch.bincount(cnt_per_edge_type))
            cnt = torch.bincount(cnt_per_edge_type)[most_freq_appeared_edge_type]

            print(f"num_edge_type: {num_edge_type}, type {most_freq_appeared_edge_type} "
                  f"appeared {cnt} times")
        else:
            ValueError("data has no attribute named edge_type")


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_num_params(model):
    return sum([p.numel() for p in model.parameters()])


# copy layer
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# masking
def masked_edge_index(edge_index, edge_mask):
    """
    :param edge_index: (2, num_edges)
    :param edge_mask: (num_edges) -- source node 기준임
    :return: masked edge_index
    """
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        # if edge_index == SparseTensor
        return masked_select_nnz(edge_index, edge_mask, layout='coo')


# distribution
def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


# fill
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)








