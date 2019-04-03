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
        return (inverse_logit(output["logits"].detach().cpu().numpy()))

    def predict(self, dataset: Iterable[Instance]) -> np.ndarray:
        predict_iterator = self.iterator(dataset, num_epochs=1, shuffle=False)

        self.model.eval()

        predict_result = []
        predict_tqdm_generator = tqdm(predict_iterator,
                                      total=self.iterator.get_num_batches(dataset))

        with torch.no_grad():
            for batch in predict_tqdm_generator:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                predict_result.append(self.__extract_data(batch))
        return np.concatenate(predict_result, axis=0)


def inverse_logit(p):
    if p > 0:
        return 1. / (1. + np.exp(-p))
    elif p <= 0:
        np.exp(p) / (1 + np.exp(p))
    else:
        raise ValueError
