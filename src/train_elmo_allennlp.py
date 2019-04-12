import os

from data.definitions import OPTION_FILE, WEIGHT_FILE, PRETRAINED_ELMO
from prepare_allennlp_data import data_iterator, dataset_reader
from tl_allennlp.base_model import BaseModel

import torch

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary

HIDDEN_DIM = 128
SEED = 42
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50


def main():
    cuda_device = -1

    torch.manual_seed(SEED)

    elmo_embedder = ElmoTokenEmbedder(OPTION_FILE, WEIGHT_FILE)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    lstm = PytorchSeq2VecWrapper(
        torch.nn.LSTM(word_embeddings.get_output_dim(),
                      HIDDEN_DIM,
                      bidirectional=True,
                      batch_first=True)
    )

    train_dataset, dev_dataset = dataset_reader(train=True, elmo=True)
    vocab = Vocabulary()

    model = BaseModel(word_embeddings=word_embeddings,
                      encoder=lstm,
                      vocabulary=vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)

    iterator = data_iterator(vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=dev_dataset,
        cuda_device=cuda_device,
        num_epochs=EPOCHS,
        patience=5
    )

    trainer.train()

    print("*******Save Model*******\n")

    output_elmo_model_file = os.path.join(PRETRAINED_ELMO, "lstm_elmo_model.bin")
    torch.save(model.state_dict(), output_elmo_model_file)


if __name__ == "__main__":
    main()
