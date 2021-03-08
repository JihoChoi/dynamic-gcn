"""
    Author: Jiho Choi 
        - https://github.com/JihoChoi
"""

print("================================")
print("module checker")
print("================================")
print("--------------------------------")

import torch;
print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)

import torch_scatter
print("torch_scatter version:", torch_scatter.__version__)

import torch_sparse
print("torch_sparse version:", torch_sparse.__version__)

import torch_geometric
print("torch_geometric version:", torch_geometric.__version__)
print("--------------------------------")


import sys
import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch_scatter import scatter_mean
from torch_scatter import scatter_max
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv

