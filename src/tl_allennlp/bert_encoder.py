import torch

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


BERT_EMBEDDER = PretrainedBertEmbedder(
    pretrained_model="bert-base-uncased",
    top_layer_only=True
)
WORD_EMBEDDING = BasicTextFieldEmbedder({"tokens": BERT_EMBEDDER},
                                        allow_unmatched_keys=True)


class BertSentencePooler(Seq2VecEncoder):
    def forward(self,
                embs: torch.tensor,
                mask: torch.tensor=None) -> torch.tensor:
        return embs[:, 0]

    def get_output_dim(self) -> int:
        return WORD_EMBEDDING.get_output_dim()
