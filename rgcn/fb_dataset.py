# from torch_geometric.datasets import RelLinkPredDataset
# 위 코드가 실행이 되지 않는다면...
import os
from typing import Optional, Callable, List

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url


class RelLinkPredDataset(InMemoryDataset):
    urls = {
        'FB15k-237': ('https://raw.githubusercontent.com/MichSchli/'
                      'RelationPrediction/master/data/FB-Toutanova')
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        assert name in ['FB15k-237']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self) -> int:
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'entities.dict', 'relations.dict', 'test.txt', 'train.txt',
            'valid.txt'
        ]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(f'{self.urls[self.name]}/{file_name}', self.raw_dir)

    def process(self):
        with open(os.path.join(self.raw_dir, 'entities.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: int(value) for value, key in lines}

        with open(os.path.join(self.raw_dir, 'relations.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            relations_dict = {key: int(value) for value, key in lines}

        kwargs = {}
        for split in ['train', 'valid', 'test']:
            with open(os.path.join(self.raw_dir, f'{split}.txt'), 'r') as f:
                lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
                src = [entities_dict[row[0]] for row in lines]
                rel = [relations_dict[row[1]] for row in lines]
                dst = [entities_dict[row[2]] for row in lines]
                kwargs[f'{split}_edge_index'] = torch.tensor([src, dst])
                kwargs[f'{split}_edge_type'] = torch.tensor(rel)

        # For message passing, we add reverse edges and types to the graph:
        row, col = kwargs['train_edge_index']
        edge_type = kwargs['train_edge_type']
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_type = torch.cat([edge_type, edge_type + len(relations_dict)])

        data = Data(num_nodes=len(entities_dict), edge_index=edge_index,
                    edge_type=edge_type, **kwargs)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save((self.collate([data])), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'