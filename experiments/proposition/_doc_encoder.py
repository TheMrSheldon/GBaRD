from typing import Literal, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, JumpingKnowledge
from torch_geometric.utils import unbatch
from transformers import DistilBertTokenizer


class ResidualConnection(torch.nn.Module):

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input)+input


class Head(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, jk: Union[None, Literal["lstm", "max"]] = None) -> None:
        super().__init__()
        self.gat = GATConv(in_channels=hidden_size, out_channels=hidden_size, add_self_loops=True)
        self.gcns = nn.ModuleList([GCNConv(hidden_size, hidden_size) for _ in range(num_layers)])
        self.jk = JumpingKnowledge(mode=jk, channels=hidden_size, num_layers=1) if jk is not None else None

    def forward_update_edges(self, x: torch.Tensor, edge_index: torch.Tensor, edge_mask: torch.Tensor):
        _, ret = self.gat(x, edge_index, edge_attr=edge_mask, return_attention_weights=True)
        return ret

    def forward_message_passing(self, x: torch.Tensor, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        for gcn in self.gcns:
            outputs.append(gcn(outputs[-1], edge_index, edge_mask).relu())
        if self.jk is None:
            return outputs[-1]
        else:
            return self.jk(outputs)


class DocEncoder(LightningModule):
    def __init__(
        self,
        feature_size: int,
        dropout: float,
        topk: Union[float, int],
        steps: int = 1,
        num_heads: int = 1,
        num_layers: int = 2,
        jumping_knowledge: Union[None, Literal["lstm", "max"]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = 2 * feature_size
        self.feature_size = feature_size
        self.topk = topk
        self.steps = steps
        self.num_heads = num_heads
        self.initial = GCNConv(in_channels=self.feature_size, out_channels=self.hidden_size)
        self.heads = nn.ModuleList([Head(self.hidden_size, num_layers, jumping_knowledge) for _ in range(num_heads)])
        self.readout = nn.Sequential(
             nn.Dropout(dropout),
             nn.Linear(num_heads * self.hidden_size, self.feature_size),
        )
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def _apply_hardmask(
        self, edge_index: torch.Tensor, edge_weights: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Note that this implemenation only makes sense edge_index and edge_weights do not represent a batched graph
        top = self.topk if isinstance(self.topk, int) else int(self.topk * edge_index.shape[1])
        # Assuming edge_weights has shape [num_edges, 1]
        _, ind = edge_weights.flatten().topk(top, sorted=False)
        return edge_index[:, ind], edge_weights[ind]

    def _message_passing(self, x, edge_index, edge_mask: list) -> torch.Tensor:
        return torch.stack([head.forward_message_passing(x[i], edge_index[i], edge_mask[i]) for i, head in enumerate(self.heads)])

    def _update_graph_structure(self, x, edge_index) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        edge_index = [edge_index] * self.num_heads
        edge_mask = [None] * self.num_heads
        for _ in range(self.steps):
            for i, head in enumerate(self.heads):
                edge_index[i], edge_mask[i] = head.forward_update_edges(x[i], edge_index[i], edge_mask[i])
            x = self._message_passing(x, edge_index, edge_mask)
            if not self.training:  # on inference we want to hard mask
                for i in range(self.num_heads):
                    edge_index[i], edge_mask[i] = self._apply_hardmask(edge_index[i], edge_mask[i])
        return x, edge_index, edge_mask

    def _compute_graph_embedding(self, x: torch.Tensor, edge_index, edge_mask, batch) -> torch.Tensor:
        # We can't do pooling here since ColBERT's late interaction will require (batch, words, feature_size) tensors
        # but pooling would result in a (batch, feature_size) tensor
        # return global_mean_pool(x, batch)  # (batch, feature_size)
        num_words = x.size(1)
        x = self.readout(x.transpose(0, 1).reshape((num_words, -1)))
        assert x.shape == (num_words, self.feature_size)
        return pad_sequence(unbatch(x, batch), batch_first=True)  # (batch, words, feature_size)

    def forward(self, x: torch.Tensor, edge_index, batch) -> tuple[torch.FloatTensor, Data]:
        x = self.initial(x, edge_index).relu()
        x = x.expand(self.num_heads, -1, -1)
        x, edge_index, edge_mask = self._update_graph_structure(x, edge_index)
        emb = self._compute_graph_embedding(x, edge_index, edge_mask, batch)
        return emb, Data(x=x, edge_index=edge_index, edge_weight=edge_mask, batch=batch)
