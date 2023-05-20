from hashlib import sha1

import torch
from torch_geometric.utils import degree


def tensor_hash(tensor: torch.Tensor) -> str:
    bytes = tensor.cpu().numpy().tobytes()
    return sha1(bytes, usedforsecurity=False).hexdigest()


def unbatch_edge_attr(edge_attr: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> list[torch.Tensor]:
    """
    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 3, 4, 5],
                                       [1, 0, 2, 5, 5, 3]])
        >>> edge_attr = torch.tensor([[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]])
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_attr(edge_attr, edge_index, batch)
        (tensor([[0,1], [2,3], [4,5]]),
         tensor([[6,7], [8,9], [10,11]]))
    """
    edge_batch = batch[edge_index[0]]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_attr.split(sizes, dim=0)
