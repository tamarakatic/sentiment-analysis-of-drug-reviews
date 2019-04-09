import os
import torch

from prepare_allennlp_data import data_iterator, dataset_reader
from tl_allennlp.base_model import BaseModel
from tl_allennlp.definitions import PRETRAINED_BERT

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import Trainer

HIDDEN_DIM = 128
SEED = 42
LEARNING_RATE = 3e-4
EPOCHS = 20
BATCH_SIZE = 32

BERT_EMBEDDER = PretrainedBertEmbedder(
    pretrained_model="bert-base-uncased",
    top_layer_only=True
)
WORD_EMBEDDINGS = BasicTextFieldEmbedder({"tokens": BERT_EMBEDDER},
                                         allow_unmatched_keys=True)


class BertSentencePooler(Seq2VecEncoder):
    def forward(self,
                embs: torch.tensor,
                mask: torch.tensor=None) -> torch.tensor:
        return embs[:, 0]

    def get_output_dim(self) -> int:
        return WORD_EMBEDDINGS.get_output_dim()


def main():
    cuda_gpu = torch.cuda.is_available()
    torch.manual_seed(SEED)

    train_dataset, dev_dataset = dataset_reader(train=True, elmo=False)
    vocab = Vocabulary()

    encoder = BertSentencePooler(vocab)

    model = BaseModel(word_embeddings=WORD_EMBEDDINGS,
                      encoder=encoder,
                      vocabulary=vocab)

    model.cuda() if cuda_gpu else model

    iterator = data_iterator(vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=dev_dataset,
        cuda_device=0 if cuda_gpu else -1,
        num_epochs=EPOCHS,
        patience=5
    )

    trainer.train()

    print("*******Save Model*******\n")

    output_bert_model_file = os.path.join(PRETRAINED_BERT, "bert_model.bin")
    torch.save(model.state_dict(), output_bert_model_file)


if __name__ == '__main__':
    main()
