from data.definitions import OPTION_FILE, WEIGHT_FILE, PRETRAINED_ELMO
import torch
import os

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedders
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.data.vocabulary import Vocabulary
from elmo.base_model import BaseModel

HIDDEN_DIM = 128


def eval_elmo():
    elmo_embedders = ElmoTokenEmbedders(OPTION_FILE, WEIGHT_FILE)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedders})

    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(word_embeddings.get_output_dim(),
                      HIDDEN_DIM,
                      bidirectional=True,
                      batch_first=True)
    )

    # _, _, test_dataset = dataset_reader()
    vocabulary = Vocabulary()

    model = BaseModel(word_embeddings=word_embeddings,
                      encoder=encoder,
                      vocabulary=vocabulary)

    output_elmo_model_file = os.path.join(PRETRAINED_ELMO, "lstm_elmo_model.bin")
    model.load_state_dict(output_elmo_model_file)
    model.eval()
    model.cuda()
    import ipdb
    ipdb.set_trace()  # XXX Breakpoint


if __name__ == '__main__':
    eval_elmo()
