import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Optional, Iterator


from allennlp.data.fields import TextField, ArrayField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import Instance
from allennlp.data.tokenizers import Token


MAX_SEQ_LENGTH = 128
CLASSES = [0, 1, 2]


class ElmoDatasetReader(DatasetReader):

    def __init__(self,
                 tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_length: Optional[int]=MAX_SEQ_LENGTH) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_length = max_seq_length

    def text_to_instance(self,
                         tokens: List[Token],
                         idx: int=None,
                         label: np.ndarray=None,
                         nclasses: int=None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        id_field = MetadataField(idx)
        fields["id"] = id_field

        label_field = [0 if label != idx else 1 for idx in range(nclasses)]
        fields["label"] = ArrayField(array=label_field)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)

        for (i, row) in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["review"])],
                row["Index"],
                row["rating"],
                len(CLASSES)
            )
