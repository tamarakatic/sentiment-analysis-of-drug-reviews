import torch

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

OUTPUT_DIM = 128


class BertSentencePooler(Seq2VecEncoder):
    def forward(self,
                embs: torch.tensor,
                mask: torch.tensor=None) -> torch.tensor:
        return embs[:, 0]

    def get_output_dim(self) -> int:
        return OUTPUT_DIM
