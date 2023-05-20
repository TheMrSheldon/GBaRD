# https://github.com/mrjleo/ranking-models/blob/master/models/bert.py
from typing import Any, Iterable, Tuple

import torch
import transformers
from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor
from transformers import BertModel, DistilBertTokenizer, get_constant_schedule_with_warmup

from ._colbert import ColBERT

Input = Tuple[str, str]
Batch = Tuple[dict, dict]


class MMSEColBERTProcessor(DataProcessor):

    def __init__(self, char_limit: int, append_mask: int=8) -> None:
        super().__init__()
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.char_limit = char_limit
        self.append_mask = append_mask

        # without this, there will be a message for each tokenizer call
        transformers.logging.set_verbosity_error()

    def get_model_input(self, query: str, doc: str) -> Input:
        # empty queries or documents might cause problems later on
        if len(query.strip()) == 0:
            query = "(empty)"
        if len(doc.strip()) == 0:
            doc = "(empty)"

        # limit characters to avoid tokenization bottlenecks
        return query+" [MASK]" * self.append_mask, doc

    def get_model_batch(self, inputs: Iterable[Input]) -> Batch:
        queries, docs = zip(*inputs)
        doc_in = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        query_in = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        return (query_in, doc_in)


class MMSEColBERTRanker(Ranker):
    def __init__(self, lr: float, warmup_steps: int, hparams: dict[str, Any],) -> None:
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(hparams)

        self.colbert: ColBERT = ColBERT.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco", return_dict=True)
        for p in self.colbert.parameters():
            p.requires_grad = not hparams["freeze_bert"]

    def forward(self, batch: Batch) -> torch.Tensor:
        query, doc = batch
        return self.colbert(query=query, document=doc)

    def configure_optimizers(self) -> Tuple[list[Any], list[Any]]:
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
