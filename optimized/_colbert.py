# https://github.com/sebastian-hofstaetter/neural-ranking-kd/blob/main/minimal_colbert_usage_example.ipynb
import torch
from transformers import AutoModel, PretrainedConfig, PreTrainedModel


class ColBERTConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True


class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    We use a dot-product instead of cosine per term (slightly better)
    """

    config_class = ColBERTConfig
    base_model_prefix = "bert_model"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)

    def forward(self, query: dict[str, torch.LongTensor], document: dict[str, torch.LongTensor]):
        query_vecs = self.forward_representation(query)
        document_vecs = self.forward_representation(document)

        score = self.forward_aggregation(query_vecs, document_vecs, query["attention_mask"], document["attention_mask"])
        return score

    def forward_representation(self, tokens, sequence_type=None) -> torch.Tensor:
        vecs = self.bert_model(**tokens)[0]  # assuming a distilbert model here
        vecs = self.compressor(vecs)

        # if encoding only, zero-out the mask values so we can compress storage
        if sequence_type == "doc_encode" or sequence_type == "query_encode":
            vecs = vecs * tokens["tokens"]["mask"].unsqueeze(-1)

        return vecs

    def forward_aggregation(self, query_vecs, document_vecs, query_mask, document_mask):
        # create initial term-x-term scores (dot-product)
        score = torch.bmm(query_vecs, document_vecs.transpose(2, 1))

        # mask out padding on the doc dimension (mask by -1000, because max should not select those, setting it to 0
        # might select them)
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~exp_mask] = -10000

        # max pooling over document dimension
        score = score.max(-1).values

        # mask out paddding query values
        score[~(query_mask.bool())] = 0

        # sum over query values
        score = score.sum(-1)

        return score
