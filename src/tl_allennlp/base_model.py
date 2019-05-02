from typing import Dict, Any

from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data.vocabulary import Vocabulary

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
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                id: Any,
                label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

        output = {"logits": logits}
        output["loss"] = self.loss(logits, label)

        return output
