from pathlib import Path
from typing import Any, Iterable, Literal, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from ranking_utils.model import Ranker, TrainingMode
from torch.nn import BCEWithLogitsLoss, Module, MSELoss
from torch_geometric.data import Data

from transformers import get_constant_schedule_with_warmup

from common.utils import tensor_hash

from ._colbert import ColBERT
from ._doc_encoder import DocEncoder
from ._processor import Batch


class Representation(NamedTuple):
    vector: torch.Tensor
    attention_mask: torch.Tensor
    graph: Optional[Data] = None


class LinearMSELoss(Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = MSELoss()
        # self.linear: Optional[torch.nn.Linear] = None
        self.linear = torch.nn.Linear(768, 768, bias=False)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inp = input.view((-1, input.size(-1)))
        tgt = target.view((-1, target.size(-1)))
        # if self.linear is None:
        #    self.linear = torch.nn.Linear(inp.size(-1), tgt.size(-1), bias=False)
        return self.mse(self.linear(inp), tgt)


class MMSEColBERTRanker(Ranker):
    def __init__(self) -> None:
        super().__init__(training_mode=TrainingMode.CONTRASTIVE)
        self.colbert: ColBERT = ColBERT.from_pretrained(
            "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        )

    def forward(self, batch: Batch) -> torch.FloatTensor:
        return self.colbert.forward(batch.queries, batch.docs)


class ProposedRanker(Ranker):
    def __init__(
        self,
        lr: float,
        warmup_steps: int,
        alpha: float = 0.5,
        sparsity_tgt: float = 3,
        topk: float = 1.0,
        dropout: float = 0.5,
        iteration_steps: int = 1,
        num_heads: int = 3,
        num_layers: int = 2,
        jumping_knowledge: Union[None, Literal["lstm", "max"]] = None,
        cache_dir: Union[str, Path, None] = "./cache/colbert/",
    ) -> None:
        super().__init__(training_mode=TrainingMode.CONTRASTIVE)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.sparsity_tgt = sparsity_tgt
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.doc_encoder = DocEncoder(feature_size=768, dropout=dropout, topk=topk, steps=iteration_steps, num_heads=num_heads, num_layers=num_layers, jumping_knowledge=jumping_knowledge)
        self.colbert: ColBERT = ColBERT.from_pretrained(
            "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        )
        self.mseloss = MSELoss()
        self.bce = BCEWithLogitsLoss(reduction="none")
        self.save_hyperparameters(ignore=["cache_dir"])

        self.hidden_loss = LinearMSELoss()

        # Freeze colbers parameters since we only want to train the doc_encoder for now
        for p in self.colbert.parameters():
            p.requires_grad = False

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _fw_colbert_single(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input = {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0)}
        return self.colbert.forward_representation(input).squeeze()

    @torch.no_grad()
    def _fw_colbert_single_or_load_from_cache(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # We only use unmasked entries of the input_ids to compute a hash for the input. Otherwise we may run into the
        # following issue:
        # Let tokenized inputs be of different dimensions i.e. the following input_ids:
        # id1: [ 101, 7592, 2088,  102]    and    id2: [ 101, 3231,  102]
        # When batched id2 will be extended to [ 101, 3231,  102, 0] to match the dimensions of id1 but these 0's are
        # masked in the attention masks:
        # am1: [ 1, 1, 1, 1]    and    am2: [ 1, 1, 1, 0]
        # In another batch, however, the same input as for id2 would have input_ids: [ 101, 3231,  102, 0, 0] with the
        # last two entries masked out and a simple hash of the input_ids would not recognize it to be effectively the
        # same input.
        input_length = torch.nonzero(attention_mask)[-1] + 1
        unmasked = input_ids[:input_length]
        if not self.cache_dir:
            data = self._fw_colbert_single(unmasked, attention_mask[:input_length])
        else:
            key = tensor_hash(unmasked)
            cache_file = self.cache_dir / f"{key}"
            if cache_file.exists():
                data = torch.load(cache_file, map_location=self.device)
                assert isinstance(data, torch.Tensor)
            else:
                data = self._fw_colbert_single(unmasked, attention_mask[:input_length])
                torch.save(data, cache_file)
        data = F.pad(input=data, pad=(0, 0, 0, attention_mask.size(0) - input_length), mode="constant", value=0)
        assert data.size(0) == input_ids.size(0)
        return data

    @torch.no_grad()
    def _forward_colbert_representation_or_load_from_cache(self, input: dict[str, torch.LongTensor]) -> torch.Tensor:
        out = torch.stack(
            [
                self._fw_colbert_single_or_load_from_cache(*pair)
                for pair in zip(input["input_ids"], input["attention_mask"])
            ]
        )
        return out

    def _sparsity(self, doc_graphs: Data) -> torch.Tensor:
        # 1) Calculate the l2-norm for each graph's edge_weights
        #norms = []
        #for weight in doc_graphs.edge_weight:
        #    edge_attrs = unbatch_edge_attr(weight, doc_graphs.edge_index, doc_graphs.batch)
        #    norms.append(torch.stack([vector_norm(v, ord=2) for v in edge_attrs]))
        #return torch.mean(torch.stack(norms).float(), dim=0)
        # 2) Frobenius Norm of adjacency matrix
        batch_size = int(doc_graphs.batch.max()+1)
        num_heads = len(doc_graphs.edge_weight)
        square_sum = torch.cat(doc_graphs.edge_weight).pow(2).sum()
        return square_sum/(batch_size*num_heads)

    def _enc_doc(self, input: tuple[Data, torch.Tensor]) -> Representation:
        doc_graphs, docs = input
        doc_vecs, new_graphs = self.doc_encoder(doc_graphs.x, doc_graphs.edge_index, doc_graphs.batch)
        return Representation(vector=doc_vecs, attention_mask=docs["attention_mask"], graph=new_graphs)

    def _enc_query(self, queries: dict[str, torch.LongTensor]) -> Representation:
        query_vecs = self._forward_colbert_representation_or_load_from_cache(queries)
        return Representation(vector=query_vecs, attention_mask=queries["attention_mask"])

    def _late_interaction(self, query: Representation, doc: Representation) -> torch.Tensor:
        return self.colbert.forward_aggregation(query.vector, doc.vector, query.attention_mask, doc.attention_mask)

    def forward(
        self, batch: Batch, return_all: bool = False
    ) -> Union[torch.FloatTensor, tuple[Representation, Representation]]:
        query_rep = self._enc_query(batch.queries)
        doc_rep = self._enc_doc((batch.doc_graphs, batch.docs))
        classlbl = self._late_interaction(query_rep, doc_rep)
        if return_all:
            return classlbl, (query_rep, doc_rep)
        return classlbl

    def training_step(self, batch) -> torch.Tensor:
        assert self.training

        pos_batch, _, _ = batch  # We expect to only get positives here
        assert isinstance(pos_batch, Batch)
        pred, (query_rep, doc_rep) = self(pos_batch, return_all=True)
        assert isinstance(query_rep, Representation)
        assert isinstance(doc_rep, Representation)

        # Compute classification of the sampled negatives for contrastive loss
        # Due to issues with gradient calculation we will only perform in-batch negatives instead of cross-batch
        # negatives
        batchsize = query_rep.vector.size(0)
        num_queries = query_rep.vector.size(0)
        # [d1, d2, d3] --> [d1, d1, d2, d2, d3, d3]
        neg_batch = Representation(
            vector=(
                doc_rep.vector.unsqueeze(1)  # (batch, 1, maxtokens_d, 768)
                .expand(-1, num_queries - 1, -1, -1)  # (batch, numqueries-1, maxtokens_d, 768)
                .reshape((-1, *doc_rep.vector.shape[-2:]))  # (batch * (numqueries-1), maxtokens_d, 768)
            ),
            attention_mask=(
                doc_rep.attention_mask.unsqueeze(1)  # (batch, 1, maxtokens_d)
                .expand(-1, num_queries - 1, -1)  # (batch, numqueries-1, maxtokens_d)
                .reshape((-1, doc_rep.attention_mask.shape[-1]))  # (batch * (numqueries-1), maxtokens_d)
            ),
        )
        # [q1, q2, q3] --> [q1, q2, q3, q1, q2, q3, q1, q2, q3]
        kept_indices = [i for i in range(num_queries * batchsize) if i % (num_queries + 1) != 0]
        neg_queries = Representation(
            vector=(
                query_rep.vector.expand(  # (batch, maxtokens_q, 768)
                    num_queries, -1, -1, -1
                ).reshape(  # (numqueries, batch, maxtokens_q, 768)
                    (-1, *query_rep.vector.shape[-2:])
                )[  # (batch*numqueries, maxtokens_q, 768)
                    kept_indices
                ]  # (batch*(numqueries-1), maxtokens_q, 768)
            ),
            attention_mask=(
                query_rep.attention_mask.expand(  # (batch, maxtokens_q)
                    num_queries, -1, -1
                ).reshape(  # (numqueries, batch, maxtokens_q)
                    (-1, query_rep.attention_mask.shape[-1])
                )[  # (batch*numqueries, maxtokens_q)
                    kept_indices
                ]  # (batch*(numqueries-1), maxtokens_q)
            ),
        )
        assert neg_batch.vector.size(0) == batchsize * (num_queries - 1)
        assert neg_batch.attention_mask.size(0) == batchsize * (num_queries - 1)
        assert neg_queries.vector.size(0) == batchsize * (num_queries - 1)
        assert neg_queries.attention_mask.size(0) == batchsize * (num_queries - 1)

        neg_outputs = self._late_interaction(neg_queries, neg_batch)
        pos_outputs = pred

        # We try different options for the classification loss
        # 1) Contrastive Loss
        # neg_exp = torch.exp(neg_outputs)  # (batch*(numqueries-1))
        # neg_sum = neg_exp.reshape((num_queries, num_queries-1)).sum(1)  # (batch)
        # pos_exp = torch.exp(pos_outputs)
        # classification_loss = torch.mean(-torch.log(pos_exp / (pos_exp+neg_sum)).flatten())
        # 2) Cross Entropy loss
        out = torch.cat((pos_outputs, neg_outputs))
        labels = torch.cat(
            (torch.ones_like(pos_outputs, device=self.device), torch.zeros_like(neg_outputs, device=self.device))
        )
        classification_loss = torch.mean(self.bce(out, labels))

        # Compute teacher embedding for the distillation loss
        teacher_doc_vecs = self._forward_colbert_representation_or_load_from_cache(pos_batch.docs)
        teacher_doc_rep = Representation(teacher_doc_vecs, pos_batch.docs["attention_mask"])
        teacher_neg_doc_rep = Representation(
            vector=(
                teacher_doc_rep.vector.unsqueeze(1)  # (batch, 1, maxtokens_d, 768)
                .expand(-1, num_queries - 1, -1, -1)  # (batch, numqueries-1, maxtokens_d, 768)
                .reshape((-1, *teacher_doc_rep.vector.shape[-2:]))  # (batch * (numqueries-1), maxtokens_d, 768)
            ),
            attention_mask=(
                teacher_doc_rep.attention_mask.unsqueeze(1)  # (batch, 1, maxtokens_d)
                .expand(-1, num_queries - 1, -1)  # (batch, numqueries-1, maxtokens_d)
                .reshape((-1, teacher_doc_rep.attention_mask.shape[-1]))  # (batch * (numqueries-1), maxtokens_d)
            ),
        )
        # We try different options for the distillation loss
        # 1) simple MSE loss
        # distillation_loss = torch.mean(self.mseloss(doc_rep.vector, teacher_doc_vecs))
        # 2) TinyBERT's L_hidn (MSE with additional linear transformation)
        distillation_loss = self.hidden_loss(doc_rep.vector, teacher_doc_vecs)
        # 3) KL-Divergence
        # out = pos_outputs
        # labels = self._late_interaction(query_rep, teacher_doc_rep)
        # distillation_loss += torch.mean(F.kl_div(F.log_softmax(out), F.log_softmax(labels), log_target=True))
        # 4) TinyBERT's L_pred (Cross Entropy Loss in Eq. 10)
        # pos_labels = self._late_interaction(query_rep, teacher_doc_rep)
        # neg_labels = self._late_interaction(neg_queries, teacher_neg_doc_rep)
        # out = torch.cat((pos_outputs, neg_outputs))
        # labels = torch.cat((pos_labels, neg_labels))
        # distillation_loss += F.binary_cross_entropy_with_logits(out, -F.log_softmax(labels), reduction="mean")
        # 5) MMSE Loss
        # pos_labels = self._late_interaction(query_rep, teacher_doc_rep)
        # neg_labels = self._late_interaction(neg_queries, teacher_neg_doc_rep)
        # distillation_loss += self.mseloss(pos_outputs-neg_outputs, pos_labels-neg_labels)

        # Compute sparsity of the document representation
        # sparsity = torch.mean(self._sparsity(doc_rep.graph))
        # Additionally push sparsity towards a reasonable value (we arbitrarily chose 3)
        # sparsity = 0.1 * torch.square(torch.mean(self._sparsity(doc_rep.graph) - self.sparsity_tgt))
        sparsity = self._sparsity(doc_rep.graph)

        # Other options for losses:
        # normalize the loss
        # kl divergence loss

        # Ablation: different combinations of these lossfunctions and their effect on training
        # 1) Simple sum:
        # loss = distillation_loss + sparsity + classification_loss
        # 2) Weighted:
        loss = self.alpha * distillation_loss + (1 - self.alpha) * (0.5 * classification_loss + 0.5 * sparsity)
        # 3) Distillation loss only:
        # loss = distillation_loss
        # 4) Classification loss only:
        # loss = classification_loss

        self.log_dict(
            {
                "loss/total": loss,
                "loss/distillation": distillation_loss,
                "loss/sparsity": sparsity,
                "loss/classification": classification_loss,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Process a validation batch. The returned query IDs are internal IDs.

        Args:
            batch (ValTestBatch): A validation batch.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, torch.Tensor]: Query IDs, scores and labels.
        """
        assert not self.training
        # model_batch, q_ids, labels = batch
        # pred = self(model_batch).flatten()
        # return self.bce(pred, labels.float().flatten())
        pos_batch, _, _ = batch  # We expect to only get positives here
        assert isinstance(pos_batch, Batch)
        pred, (query_rep, doc_rep) = self(pos_batch, return_all=True)
        assert isinstance(query_rep, Representation)
        assert isinstance(doc_rep, Representation)

        # Compute classification of the sampled negatives for contrastive loss
        # Due to issues with gradient calculation we will only perform in-batch negatives instead of cross-batch
        # negatives
        batchsize = query_rep.vector.size(0)
        num_queries = query_rep.vector.size(0)
        # [d1, d2, d3] --> [d1, d1, d2, d2, d3, d3]
        neg_batch = Representation(
            vector=(
                doc_rep.vector.unsqueeze(1)  # (batch, 1, maxtokens_d, 768)
                .expand(-1, num_queries - 1, -1, -1)  # (batch, numqueries-1, maxtokens_d, 768)
                .reshape((-1, *doc_rep.vector.shape[-2:]))  # (batch * (numqueries-1), maxtokens_d, 768)
            ),
            attention_mask=(
                doc_rep.attention_mask.unsqueeze(1)  # (batch, 1, maxtokens_d)
                .expand(-1, num_queries - 1, -1)  # (batch, numqueries-1, maxtokens_d)
                .reshape((-1, doc_rep.attention_mask.shape[-1]))  # (batch * (numqueries-1), maxtokens_d)
            ),
        )
        # [q1, q2, q3] --> [q1, q2, q3, q1, q2, q3, q1, q2, q3]
        kept_indices = [i for i in range(num_queries * batchsize) if i % (num_queries + 1) != 0]
        neg_queries = Representation(
            vector=(
                query_rep.vector.expand(  # (batch, maxtokens_q, 768)
                    num_queries, -1, -1, -1
                ).reshape(  # (numqueries, batch, maxtokens_q, 768)
                    (-1, *query_rep.vector.shape[-2:])
                )[  # (batch*numqueries, maxtokens_q, 768)
                    kept_indices
                ]  # (batch*(numqueries-1), maxtokens_q, 768)
            ),
            attention_mask=(
                query_rep.attention_mask.expand(  # (batch, maxtokens_q)
                    num_queries, -1, -1
                ).reshape(  # (numqueries, batch, maxtokens_q)
                    (-1, query_rep.attention_mask.shape[-1])
                )[  # (batch*numqueries, maxtokens_q)
                    kept_indices
                ]  # (batch*(numqueries-1), maxtokens_q)
            ),
        )
        assert neg_batch.vector.size(0) == batchsize * (num_queries - 1)
        assert neg_batch.attention_mask.size(0) == batchsize * (num_queries - 1)
        assert neg_queries.vector.size(0) == batchsize * (num_queries - 1)
        assert neg_queries.attention_mask.size(0) == batchsize * (num_queries - 1)

        neg_outputs = self._late_interaction(neg_queries, neg_batch)
        pos_outputs = pred

        out = torch.cat((pos_outputs, neg_outputs))
        labels = torch.cat(
            (torch.ones_like(pos_outputs, device=self.device), torch.zeros_like(neg_outputs, device=self.device))
        )
        return torch.mean(self.bce(out, labels))

    def validation_step_end(self, step_results: dict[str, torch.Tensor]) -> None:
        """Update the validation metrics.

        Args:
            step_results (Dict[str, torch.Tensor]): Results from a validation step.
        """
        pass

    def validation_epoch_end(self, val_results: Iterable[torch.Tensor]) -> None:
        """Compute validation metrics.

        Args:
            val_results (Iterable[Dict[str, torch.Tensor]]): Results of the validation steps.
        """
        self.log("val_loss", torch.mean(torch.stack(list(val_results))), sync_dist=True)

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
