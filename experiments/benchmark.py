#! /usr/bin/env python3

"""
Usage: benchmark.py [path-to-dataset] [Proposal | MMSEColBERT]

Benchmarks the specified model on TREC DL'19 Passage. If no model was specified (i.e., neither Proposal nor MMSEcolBERT
was passed as an argument), both models are benchmarked. `path-to-dataset` denotes the path the dataset is located at.
If it does not exist, the path will be created and the dataset will be downloaded.

The benchmark is only performed on the document encoder since the query encoder and interaction function of both models
are identical.
"""

from typing import Any, Callable, Iterable

import timeit
from tqdm import tqdm
import torch
import os
import sys

from proposal._colbert import ColBERT
from transformers import DistilBertModel, DistilBertTokenizerFast as DistilBertTokenizer
from torch_geometric.data import Batch as tgBatch, Data

from codecarbon import OfflineEmissionsTracker

from common.datasets.trec19passage import TREC2019Passage

from proposal import ProposedRanker


def get_initial_retrieval(doc_encode: Callable[[list[str]], Any] = lambda x:x) -> Iterable[tuple[str, Any]]:
    dataset = TREC2019Passage(None, sys.argv[1], batch_size=40, num_workers=1)
    dataset.prepare_data()  # Download files
    num = dataset.num_top1000()
    for q, docs in tqdm(dataset.top1000(), total=num):
        yield q, doc_encode(docs[:400])


class Model:
    def __init__(self) -> None:
        self.prep = 0
        self.gc = 0
        self.enc = 0

    def doc_encode(self, docs: list[str]):
        pass


@torch.inference_mode()
def benchmark(model: Model):
    print(f"Benchmarking: {type(model).__name__}", flush=True)
    doc_acc = 0
    doc_etrack = OfflineEmissionsTracker(country_iso_code="DEU", log_level='error', gpu_ids=os.environ.get('CUDA_VISIBLE_DEVICES', None))
    doc_etrack.start()
    for _, doc in get_initial_retrieval():
        start = timeit.default_timer()
        _ = model.doc_encode(doc)
        stop = timeit.default_timer()
        doc_acc += stop-start

    doc_etrack.stop()
    print(f"Results:\n\t: proc {model.prep:0.2f}s gc {model.gc:0.2f}s enc {model.enc:0.2f}s tot {doc_acc:0.2f}s", flush=True)
    print(doc_etrack.final_emissions_data)
    return (f"proc {model.prep:0.2f}s gc {model.gc:0.2f}s enc {model.enc:0.2f}s tot {doc_acc:0.2f}s", str(doc_etrack.final_emissions_data))


device = "cpu"

if torch.cuda.is_available():
    print("Switching to CUDA tensors", flush=True)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = "cuda"

tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


class MMSEColBERT(Model):

    def __init__(self) -> None:
        super.__init__()
        self.model = ColBERT.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco")
        self.model.eval()
        self.model.to(device)

    def doc_encode(self, docs: list[str]):
        start = timeit.default_timer()
        doc_batch = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        stop = timeit.default_timer()
        self.prep += stop-start

        start = timeit.default_timer()
        enc = self.model.forward_representation(doc_batch), doc_batch["attention_mask"]
        stop = timeit.default_timer()
        self.enc += stop-start

        return enc


class Proposal(Model):

    def __init__(self) -> None:
        super.__init__()
        self.bert: DistilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")
        window_size = 3

        self.bert.eval()
        self.bert.to(device)

        self.data = [1]*window_size*2
        self.diagidx = [*range(-window_size, 0), *range(1, window_size+1)]

        # Instantiate a WS3 1x1x2 model
        ranker = ProposedRanker(lr=0.00003, warmup_steps=3000, alpha=0.5, sparsity_tgt=3, topk=1.0, dropout=0.2, num_heads=1, iteration_steps=1, num_layers=2, cache_dir=None)
        ranker.eval()
        ranker.to(device)

        self.doc_encoder = ranker.doc_encoder

    def _create_gow(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Data:
        input_length = torch.nonzero(attention_mask)[-1] + 1
        x = embeddings[:input_length]
        size = x.size(0)
        tmp1 = torch.cat(tuple(torch.arange(0, n, device=device) for n in range(size-1, size-4, -1)))
        tmp2 = torch.cat(tuple(torch.arange(n, size, device=device) for n in range(1, 4)))
        edge_index = torch.stack((torch.cat((tmp1, tmp2)), torch.cat((tmp2, tmp1))))
        return Data(x, edge_index)

    def doc_encode(self, docs: list[str]):
        start = timeit.default_timer()
        input = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        embeddings = self.bert.embeddings(input["input_ids"])
        stop = timeit.default_timer()
        self.prep += stop-start

        start = timeit.default_timer()
        doc_graphs = tgBatch.from_data_list([self._create_gow(*pair) for pair in zip(embeddings, input["attention_mask"])])
        stop = timeit.default_timer()
        self.gc += stop-start

        start = timeit.default_timer()
        enc = self.doc_encoder(doc_graphs.x, doc_graphs.edge_index, doc_graphs.batch)
        stop = timeit.default_timer()
        self.enc += stop-start
        return enc


models = [Proposal, MMSEColBERT]

if len(sys.argv) >= 3:
    for model in models:
        if model.__name__ == sys.argv[2]:
            benchmark(model())
else:
    results = {model.__name__: benchmark(model()) for model in models}
    print(results)
