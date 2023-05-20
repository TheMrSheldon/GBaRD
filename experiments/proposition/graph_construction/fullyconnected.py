from itertools import product
from typing import Optional

import torch
from torch_geometric.data import Data
from transformers import DistilBertModel, DistilBertTokenizer

from ._graph_construction import GraphConstruction


class FullyConnected(GraphConstruction):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert: DistilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def __call__(self, input: str, device: Optional[str] = None) -> Data:
        # Calculate the embedding layer of the bert model for the node features
        input_ids = self.tokenizer(input, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        x = self.bert.embeddings(input_ids).squeeze()  # (seq_length, dim)
        num_tokens = x.size(0)

        # Create the edge_index for the fully connected graph with num_tokens nodes
        edge_index = torch.LongTensor(list(product(range(num_tokens), range(num_tokens))), device=device).T

        assert x.size() == (num_tokens, 768)
        assert edge_index.size() == (2, num_tokens**2)

        return Data(x=x, edge_index=edge_index)
