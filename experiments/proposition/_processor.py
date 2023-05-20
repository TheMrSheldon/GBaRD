from hashlib import sha1
from pathlib import Path
from typing import Iterable, NamedTuple, Union

import torch
from ranking_utils.model.data import DataProcessor
from torch_geometric.data import Batch as tgBatch
from torch_geometric.data import Data

# from torch_geometric.utils.convert import to_scipy_sparse_matrix
from transformers import DistilBertTokenizer

from .graph_construction import GraphConstruction


class Input(NamedTuple):
    doc: str
    query: str


class Batch(NamedTuple):
    docs: dict[str, torch.LongTensor]
    queries: dict[str, torch.LongTensor]
    doc_graphs: Data


class ProposedDataProcessor(DataProcessor):
    def __init__(
        self,
        graph_construction: GraphConstruction,
        query_limit: int = 10000,
        cache_dir: Union[str, Path, None] = "./cache/graphs/",
        append_mask: int = 0,
        device: Union[str, None] = None,
    ) -> None:
        super().__init__()
        assert isinstance(graph_construction, GraphConstruction), "invalid type for graph_construction"
        self.query_limit = query_limit
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.graph_construction = graph_construction
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.append_mask = append_mask
        self.device = device

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _construct_graph_or_load_from_cache(self, doc: str) -> Data:
        if self.cache_dir is None:
            return self.graph_construction(doc, device=self.device)
        # We can't use python's hash here since it is not consistent across runs
        # key = hash(doc).to_bytes(8, "big", signed=True).hex()
        key = sha1(doc.encode(), usedforsecurity=False).hexdigest()
        cache_file = self.cache_dir / f"{key}"
        if cache_file.exists():
            data = torch.load(cache_file)
            assert isinstance(data, Data)
        else:
            data = self.graph_construction(doc, device=self.device)
            torch.save(data, cache_file)

        return data

    def _construct_doc_batch(self, docs: list[str]) -> Data:
        batch = tgBatch.from_data_list([self._construct_graph_or_load_from_cache(doc) for doc in docs])
        assert isinstance(batch, Data)
        return batch

    def get_model_input(self, query: str, doc: str) -> Input:
        query = query.strip() or "(empty)"
        doc = doc.strip() or "(empty)"
        # Mimick some query expansion by adding masks to the query
        # (https://github.com/sebastian-hofstaetter/neural-ranking-kd/blob/main/minimal_colbert_usage_example.ipynb)
        return Input(doc=doc, query=query[: self.query_limit] + " [MASK]" * self.append_mask)

    def get_model_batch(self, inputs: Iterable[Input]) -> Batch:
        docs, queries = zip(*inputs)
        doc_in = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        query_in = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")

        return Batch(
            docs={"input_ids": doc_in["input_ids"], "attention_mask": doc_in["attention_mask"]},
            queries={"input_ids": query_in["input_ids"], "attention_mask": query_in["attention_mask"]},
            doc_graphs=self._construct_doc_batch([input.doc for input in inputs]),
        )
