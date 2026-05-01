# graph.py - v99
import torch
from torch_geometric.data import Data

def build_graph(window: torch.Tensor, time_k: int = 1) -> Data:
    x = window  # [T, C]
    T = x.size(0)
    edges = []
    for i in range(T):
        for j in range(max(0, i - time_k), min(T, i + time_k + 1)):
            edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)
