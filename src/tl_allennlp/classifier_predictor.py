import numpy as np
from tqdm import tqdm
from typing import Iterable

import torch

from allennlp.models import Model
from allennlp.data.iterators import DataIterator
from allennlp.data import Instance
from allennlp.nn import util as nn_util


class ClassifierPredictor:

    def __init__(self,
                 model: Model,
                 iterator: DataIterator,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def __extract_data(self, batch) -> np.ndarray:
        output = self.model(**batch)
        inv_logits = inverse_logit(output["logits"].detach().cpu().numpy())
        return inv_logits

    def predict(self, dataset: Iterable[Instance]) -> np.ndarray:
        predict_iterator = self.iterator(dataset, num_epochs=1, shuffle=False)

        self.model.eval()

        predict_result = []

        with torch.no_grad():
            for batch in predict_iterator:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                predict_result.append(self.__extract_data(batch))
        return np.concatenate(predict_result, axis=0)

    def evaluate(self, dataset):
        eval_iterator = self.iterator(dataset, num_epochs=1, shuffle=False)

        self.model.eval()

        eval_result = []

        with torch.no_grad():
            for batch in tqdm(eval_iterator, total=self.iterator.get_num_batches(dataset)):
                batch = nn_util.move_to_device(batch, self.cuda_device)
                eval_result.append(self.__extract_data(batch))
        return np.concatenate(eval_result, axis=0)


def inverse_logit(p):
    pos_ids = p > 0
    neg_ids = p <= 0
    sigmoids = np.zeros(p.shape)
    sigmoids[pos_ids] = 1.0 / (1.0 + np.exp(-p[pos_ids]))
    sigmoids[neg_ids] = np.exp(p[neg_ids]) / (1.0 + np.exp(p[neg_ids]))
    return sigmoids
