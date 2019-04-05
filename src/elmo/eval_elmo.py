import os
import torch
from data.definitions import OPTION_FILE, WEIGHT_FILE, PRETRAINED_ELMO
from elmo.prepare_elmo_data import dataset_reader
from elmo.classifier_predictor import ClassifierPredictor

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from elmo.base_model import BaseModel

HIDDEN_DIM = 128
BATCH_SIZE = 32


def load_model():
    elmo_embedders = ElmoTokenEmbedder(OPTION_FILE, WEIGHT_FILE)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedders})

    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(word_embeddings.get_output_dim(),
                      HIDDEN_DIM,
                      bidirectional=True,
                      batch_first=True)
    )

    vocabulary = Vocabulary()

    model = BaseModel(word_embeddings=word_embeddings,
                      encoder=encoder,
                      vocabulary=vocabulary)

    output_elmo_model_file = os.path.join(PRETRAINED_ELMO, "lstm_elmo_model.bin")
    model.load_state_dict(torch.load(output_elmo_model_file))
    return model


if __name__ == '__main__':
    model = load_model()
    test_dataset = dataset_reader(train=False)

    basic_iterator = BasicIterator(batch_size=BATCH_SIZE)
    basic_iterator.index_with(model.vocab)

    cuda_device = -1
    predictor = ClassifierPredictor(model, basic_iterator, cuda_device=cuda_device)
    predictor.evaluate(test_dataset)
