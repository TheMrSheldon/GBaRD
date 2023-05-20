from typing import Optional, Union

from torch import Tensor, arange, cat, no_grad, stack
from torch.nn import Dropout, Linear, Module, Sequential
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import unbatch
from transformers import (
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.distilbert.modeling_distilbert import (
    Embeddings as DBEmbeddings,
)

from ._colbert import ColBERT, ColBERTConfig


class GoWConstruction(Module):
    def __init__(self, window_size: int = 3) -> None:
        super().__init__()
        self.window_size = window_size

        dbconf = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.embeddings = DBEmbeddings(dbconf)

        # To load the embeddings from pre-trained DistilBERT
        # bert: DistilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.embeddings = bert.embeddings

    def _create_gow(self, embeddings: Tensor, length: int) -> Data:
        device = embeddings.device
        x = embeddings[:length]
        tmp1 = cat(tuple(arange(0, length - n, device=device) for n in range(1, 1 + self.window_size)))
        tmp2 = cat(tuple(arange(n, length, device=device) for n in range(1, 1 + self.window_size)))
        edge_index = stack((cat((tmp1, tmp2)), cat((tmp2, tmp1))))
        return Data(x, edge_index)

    @no_grad()
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Data:
        embeddings = self.embeddings(input_ids)
        lengths = attention_mask.sum(dim=1)
        return Batch.from_data_list([self._create_gow(*pair) for pair in zip(embeddings, lengths)])


class GraphToEmbedding(Module):
    def __init__(self, feature_size: int, dropout: float) -> None:
        super().__init__()
        hidden_size = 2 * feature_size
        self.feature_size = feature_size

        self.gat = GATConv(in_channels=hidden_size, out_channels=hidden_size, add_self_loops=True)
        self.gcn1 = GCNConv(hidden_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

        self.initial = GCNConv(in_channels=self.feature_size, out_channels=hidden_size)
        self.readout = Sequential(
            Dropout(dropout),
            Linear(hidden_size, self.feature_size),
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.initial(x, edge_index).relu()
        # Update Graph Structure
        _, (edge_index, edge_mask) = self.gat(x, edge_index, return_attention_weights=True)
        x = self.gcn1(x, edge_index, edge_mask).relu()
        x = self.gcn2(x, edge_index, edge_mask).relu()

        # Compute embedding
        # (num_words, hidden_size) -> (num_words, feature_size) -> (batch, num_words, feature_size)
        emb = pad_sequence(unbatch(self.readout(x), batch), batch_first=True)
        return emb


class GBaRDConfig(PretrainedConfig):
    model_type = "GBaRD"
    dropout: float = 0.0
    window_size: int = 3
    mask_expansion: int = 0


class GBaRD(PreTrainedModel):
    config_class = GBaRDConfig

    def __init__(
        self, config: GBaRDConfig, fast_tokenizer: bool = True, tokenization_device: Optional[str] = None
    ) -> None:
        super().__init__(config)
        cbconfig = ColBERTConfig.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco")
        self.colbert: ColBERT = ColBERT(cbconfig)

        # To load MMSEColBERT from pre-trained weights instead:
        # self.colbert: ColBERT = ColBERT.from_pretrained(
        #     "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        # )
        for p in self.colbert.parameters():
            p.requires_grad = False

        self.mask_expansion = config.mask_expansion
        self.construction = GoWConstruction(window_size=config.window_size)
        self.graphembedding = GraphToEmbedding(feature_size=self.colbert.config.compression_dim, dropout=config.dropout)

        TokenizerType = DistilBertTokenizerFast if fast_tokenizer else DistilBertTokenizer
        self.tokenizer: TokenizerType = TokenizerType.from_pretrained("distilbert-base-uncased")
        self.tokdevice = tokenization_device

    def forward(self, queries: Union[str, list[str]], docs: Union[str, list[str]]) -> Tensor:
        if isinstance(docs, str):
            docs = [docs]
        if isinstance(queries, str):
            queries = [queries]
        assert len(queries) == 1 or len(queries) == len(docs), f"The number of queries and documents must be the same. Got: {len(queries)} and {len(docs)}"
        
        # Tokenize queries and documents in a single batch
        queries = [q + " [MASK]" * self.mask_expansion for q in queries]
        qtoks = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        dtoks = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        qids, docids = qtoks["input_ids"].to(self.tokdevice), dtoks["input_ids"].to(self.tokdevice)
        qattn, docattn = qtoks["attention_mask"].to(self.tokdevice), dtoks["attention_mask"].to(self.tokdevice)

        # Encode Documents
        doc_graphs: Data = self.construction(input_ids=docids, attention_mask=docattn)
        doc_vecs: Tensor = self.graphembedding(doc_graphs.x, doc_graphs.edge_index, doc_graphs.batch)

        # Encode Queries
        q_vecs: Tensor = self.colbert.forward_representation({"input_ids": qids, "attention_mask": qattn})

        # Expand query tensors to the correct shape. This is only needed, if a single query was passed since then we
        # want to expand the queries embedding for the final aggregation step. This means that, if we the model was
        # tasked with computing the relevance of (Q, D1) and (Q, D2) we do not calculate Q's embedding multiple times.
        # The call signature would then only pass a single query as a string and multiple documents:
        # forward(Q, [D1, D2])
        q_vecs = q_vecs.expand(len(docs), -1, -1)
        qattn = qattn.expand(len(docs), -1)

        # Late Interaction
        return self.colbert.forward_aggregation(q_vecs, doc_vecs, qattn, docattn)
