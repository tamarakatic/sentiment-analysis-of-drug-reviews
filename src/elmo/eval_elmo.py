import os
import torch
from data.definitions import OPTION_FILE, WEIGHT_FILE, PRETRAINED_ELMO
from prepare_elmo_data import dataset_reader
from classifier_predictor import ClassifierPredictor

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from base_model import BaseModel

import pandas as pd
from sklearn import metrics

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
    eval_results = predictor.evaluate(test_dataset)

    y_true = [example['label'].array.argmax() for example in test_dataset]
    y_pred = eval_results.argmax(axis=1).tolist()

    f1_report = {}
    for average in ['micro', 'macro', 'weighted']:
        f1 = metrics.f1_score(y_true, y_pred, average=average)
        f1_report['f1_{}'.format(average)] = f1

    df = pd.DataFrame(f1_report)
    df.to_csv('../reports/elmo_report.csv', index=False)
