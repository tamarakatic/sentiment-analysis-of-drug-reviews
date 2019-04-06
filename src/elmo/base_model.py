from typing import Dict, Any

from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy

import torch

CLASSES = [0, 1, 2]


class BaseModel(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocabulary: Vocabulary,
                 output: int=len(CLASSES)) -> None:
        super().__init__(vocabulary)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                                      out_features=output)
        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self,
                id: Any,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        self.accuracy(logits, label)
        output["loss"] = self.loss(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}
