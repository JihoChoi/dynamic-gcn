import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Author: Jiho Choi 
# References
# - https://github.com/Qingfeng-Yao/Dynamic-GCN/blob/master/papercodes/2019-GCN-GAN/utils.py
# - https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py


class GraphDataset(Dataset):
    pass


class GraphSnapshotDataset(Dataset):
    def __init__(self, tree_dict, fold_x, data_path, snapshot_num=5,
                        lower=2, upper=100000, td_droprate=0, bu_droprate=0):
        self.fold_x = list(filter(lambda id: id in tree_dict and lower <= len(tree_dict[id]) <= upper, fold_x))
        self.tree_dict = tree_dict
        self.data_path = data_path
        self.snapshot_num = snapshot_num
        self.td_droprate = td_droprate
        self.bu_droprate = bu_droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        snapshot_data = []
        # TODO: torch.nn.ModuleList()
        for snapshot_index in range(self.snapshot_num):
            data = np.load(
                "{}/{}_{}_{}.npz".format(self.data_path, id, snapshot_index, self.snapshot_num),
                allow_pickle=True
            )
            edgeindex = data['edge_index']
            # DropEdge (ICLR 2020), TODO: DropEdge++: consider hop counts
            if self.td_droprate > 0:
                row = list(edgeindex[0])
                col = list(edgeindex[1])
                length = len(row)
                poslist = random.sample(range(length), int(
                    length * (1 - self.td_droprate)))
                poslist = sorted(poslist)
                row = list(np.array(row)[poslist])
                col = list(np.array(col)[poslist])
                new_td_edgeindex = [row, col]
            else:
                new_td_edgeindex = edgeindex
            burow = list(edgeindex[1])
            bucol = list(edgeindex[0])
            if self.bu_droprate > 0:
                length = len(burow)
                poslist = random.sample(range(length), int(length * (1 - self.bu_droprate)))
                poslist = sorted(poslist)
                row = list(np.array(burow)[poslist])
                col = list(np.array(bucol)[poslist])
                new_bu_edgeindex = [row, col]
            else:
                new_bu_edgeindex = [burow, bucol]
            data = Data(
                x=torch.tensor(data['x'], dtype=torch.float32),
                y=torch.LongTensor([int(data['y'])]),
                edge_index=torch.LongTensor(new_td_edgeindex),
                BU_edge_index=torch.LongTensor(new_bu_edgeindex),
                root=torch.LongTensor(data['root']),
                root_index=torch.LongTensor([int(data['root_index'])])
            )
            snapshot_data.append(data)
        return snapshot_data
