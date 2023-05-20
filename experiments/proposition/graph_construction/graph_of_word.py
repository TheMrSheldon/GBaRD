import math
from itertools import permutations
from typing import Optional

import torch
from torch_geometric.data import Data
from scipy.sparse import diags
from torch_geometric.utils import from_scipy_sparse_matrix
from transformers import DistilBertModel, DistilBertTokenizer

from ._graph_construction import GraphConstruction


class GraphOfWord(GraphConstruction):
    def __init__(self, window_size: int = 3, loops: bool = False, skip_edges: bool = False) -> None:
        super().__init__()
        print(f"Init GoW with [window_size: {window_size}; loops: {loops}; skip_edges: {skip_edges}]")
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert: DistilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.window_size = window_size
        self.loops = loops
        self.skip_edges = skip_edges

        self.bert.eval()

        self.data = [1]*window_size*2
        if loops:
            self.diagidx = [*range(-window_size, window_size+1)]
        else:
            self.diagidx = [*range(-window_size, 0), *range(1, window_size+1)]

    def __call__(self, input: str, device: Optional[str] = None) -> Data:
        # Fast but less flexibel implementation (does not support skip_edges)
        # Calculate the embedding layer of the bert model for the node features
        input_ids = self.tokenizer(input, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        if device:
            self.bert.to(device)
        x = self.bert.embeddings(input_ids).squeeze()  # (seq_length, dim)
        num_tokens = x.size(0)
        adj = diags(self.data, self.diagidx, shape=(num_tokens, num_tokens), format="coo")
        edge_index, _ = from_scipy_sparse_matrix(adj)
        return Data(x, edge_index)#.to(device, non_blocking=True))

        # Flexibel information
        # Calculate the embedding layer of the bert model for the node features
        input_ids = self.tokenizer(input, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        x = self.bert.embeddings(input_ids).squeeze()  # (seq_length, dim)
        num_tokens = x.size(0)

        num_gow_edges = (
            2 * self.window_size * (num_tokens - self.window_size - 1) + self.window_size**2 + self.window_size
        )
        if self.loops:
            num_gow_edges += num_tokens
        num_skips = int(math.sqrt(num_tokens))
        skip_edges = [] if not self.skip_edges else list(permutations(range(num_skips // 2, num_tokens, num_skips), 2))
        num_skip_edges = len(skip_edges)

        # print(f"tokens: {num_tokens};  skips: {num_skips}", flush=True)

        edges = torch.empty(size=(num_gow_edges + num_skip_edges, 2), dtype=torch.long)#, device="cpu", pin_memory=True)
        idx = 0
        for word in range(num_tokens):
            for i in range(max(0, word - self.window_size), min(num_tokens, word + self.window_size + 1)):
                if self.loops or i != word:
                    edges[idx][0] = word
                    edges[idx][1] = i
                    idx += 1
        assert idx == num_gow_edges, f"{idx} == {num_gow_edges} not met"

        for (s, t) in skip_edges:
            edges[idx][0] = s
            edges[idx][1] = t
            idx += 1

        assert idx == edges.size(0), f"{idx} == {edges.size(0)} not met"
        assert x.size() == (num_tokens, 768)
        # edges = edges.to(device)

        return Data(x, edges.T)
