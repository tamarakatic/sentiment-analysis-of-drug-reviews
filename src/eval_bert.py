import os
import pandas as pd
from sklearn import metrics
from collections import defaultdict

import torch

from prepare_allennlp_data import dataset_reader
from tl_allennlp.base_model import BaseModel
from tl_allennlp.definitions import PRETRAINED_BERT
from tl_allennlp.classifier_predictor import ClassifierPredictor
from tl_allennlp.bert_encoder import BertSentencePooler

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.iterators import BasicIterator

HIDDEN_DIM = 128
BATCH_SIZE = 32

BERT_EMBEDDER = PretrainedBertEmbedder(
    pretrained_model="bert-base-uncased",
    top_layer_only=True
)
WORD_EMBEDDINGS = BasicTextFieldEmbedder({"tokens": BERT_EMBEDDER},
                                         allow_unmatched_keys=True)


def load_bert_model():
    vocab = Vocabulary()

    encoder = BertSentencePooler(vocab)

    model = BaseModel(word_embeddings=WORD_EMBEDDINGS,
                      encoder=encoder,
                      vocabulary=vocab)

    output_bert_model_file = os.path.join(PRETRAINED_BERT, "bert_model.bin")
    model.load_state_dict(torch.load(output_bert_model_file))
    return model


if __name__ == '__main__':
    cuda_device = -1

    model = load_bert_model()

    test_dataset = dataset_reader(train=False, elmo=False)

    basic_iterator = BasicIterator(batch_size=BATCH_SIZE)
    basic_iterator.index_with(model.vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)

    predictor = ClassifierPredictor(model, basic_iterator, cuda_device=cuda_device)
    eval_results = predictor.evaluate(test_dataset)

    y_true = [row['label'].label for row in test_dataset]
    y_pred = eval_results.argmax(axis=1).tolist()

    f1_report = defaultdict(list)
    for average in ['micro', 'macro', 'weighted']:
        f1 = metrics.f1f1_score(y_pred, y_true, average=average)
        f1_report['f1_{}'.format(average)].append(f1)

    print("******************Bert Report***************\n{}".format(f1_report))
    df = pd.DataFrame(f1_report)
    df.to_csv('../reports/bert_report.csv', index=False)
