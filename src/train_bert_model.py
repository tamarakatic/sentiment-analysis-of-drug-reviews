import os
import numpy as np
import random
from tqdm import tqdm, trange
import logging

from sklearn.metrics import f1_score

from data.definitions import TRAIN_PATH, TEST_PATH, DEV_PATH
from data.definitions import OUTPUT_BERT_DIR, BERT_PRETRAINED_PATH

from bert.multiclass_data_processor import MultiClassDataProcessor
from bert.convert_examples_to_features import convert_examples_to_features

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

model_state_dict = None
logger = logging.getLogger(__name__)

args = {
    "no_cuda": True,
    "bert_model": "bert-base-uncased",
    "output_dir": OUTPUT_BERT_DIR,
    "cache_dir": BERT_PRETRAINED_PATH,
    "max_seq_length": 128,
    "do_train": True,
    "do_eval": True,
    "do_test": True,
    "do_lower_case": True,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "test_batch_size": 32,
    "learning_rate": 5e-5,
    "num_train_epochs": 10.0,
    "warmup_proportion": 0.1,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1
}


if __name__ == "__main__":
    if args["local_rank"] == -1 or args["no_cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args["no_cuda"] else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args["local_rank"])
        device = torch.device("cuda", args["local_rank"])
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args["local_rank"] in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args["local_rank"] != -1)))

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args["seed"])

    processor = MultiClassDataProcessor()
    label_list = processor.get_labels()
    print("Labels: {}".format(label_list))
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(
        args["bert_model"], do_lower_case=args["do_lower_case"])

    train_examples = None
    num_train_steps = None

    if args["do_train"]:
        train_examples = processor.get_train_examples(TRAIN_PATH)
        num_train_steps = int(len(train_examples) / args["train_batch_size"] /
                              args["gradient_accumulation_steps"]) * args["num_train_epochs"]

        if args["local_rank"] != -1:
            num_train_steps = num_train_steps // torch.distributed.get_world_size()

    # prepare model
    cache_dir = args["cache_dir"] if args["cache_dir"] else os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args["local_rank"]))
    model = BertForSequenceClassification.from_pretrained(args["bert_model"],
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels)
    model.to(device)

    if args["local_rank"] != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed \
                and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_group_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]

    optimizer = BertAdam(optimizer_group_parameters,
                         lr=args["learning_rate"],
                         warmup=args["warmup_proportion"],
                         t_total=num_train_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args["do_train"]:
        train_features = convert_examples_to_features(
            train_examples, label_list, args["max_seq_length"], tokenizer)
        logger.info("******** Running training ********")
        logger.info("  Number of examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args["train_batch_size"])
        logger.info("  Num steps = %d", num_train_steps)

        train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        train_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(train_input_ids, train_input_mask,
                                   train_segment_ids, train_label_ids)

        if args["local_rank"] == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args["train_batch_size"])

        model.train()

        for _ in trange(int(args["num_train_epochs"]), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                logits = model(input_ids, segment_ids, input_mask, labels=None)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()
                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    if args["do_train"] and (args["local_rank"] == -1 or torch.distributed.get_rank() == 0):
        model_to_save = model.module if hasattr(model, 'module') else model

        output_model_file = os.path.join(args["output_dir"], "pytorch_model.bin")
        output_config_file = os.path.join(args["output_dir"], "bert_config.json")

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args["output_dir"])

        model_state_dict = model_to_save.load_state_dict(torch.load(output_model_file))
        # Load a trained fine-tunes model and vocabulary
        model = BertForSequenceClassification.from_pretrained(
            args["output_dir"], num_labels=num_labels, state_dict=model_state_dict)
        tokenizer = BertTokenizer.from_pretrained(
            args["output_dir"], do_lower_case=args["do_lower_case"])

    else:
        model = BertForSequenceClassification.from_pretrained(
            args["bert_model"], num_labels=num_labels)

    model.to(device)

    if args["do_eval"] and (args["local_rank"] == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(DEV_PATH)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args["max_seq_length"], tokenizer)

        logger.info("******** Running evaluation ********")
        logger.info(" Num examples = %d", len(eval_examples))
        logger.info(" Batch size = %d", args["eval_batch_size"])

        eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                     batch_size=args["eval_batch_size"])

        model.eval()

        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        preds = np.argmax(preds, axis=1)

        eval_result = {}
        eval_result["micro"] = f1_score(y_true=eval_label_ids.numpy(),
                                        y_pred=preds, average="micro")
        loss = tr_loss / nb_tr_steps if args["do_train"] else None

        eval_result["eval_loss"] = eval_loss
        eval_result["global_step"] = global_step
        eval_result["loss"] = loss

        with open("../reports/new_bert_eval_results.txt", "w") as fp:
            logger.info("******** Eval results ********")
            for key in sorted(eval_result.keys()):
                logger.info(" %s = %s", key, str(eval_result[key]))
                fp.write("%s = %s\n" % (key, str(eval_result[key])))

    if args["do_test"] and (args["local_rank"] == -1 or torch.distributed.get_rank() == 0):
        test_examples = processor.get_test_examples(TEST_PATH)
        test_features = convert_examples_to_features(
            test_examples, label_list, args['max_seq_length'], tokenizer)

        logger.info("******** Running test ********")
        logger.info(" Num examples = %d", len(test_examples))

        test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.long)

        test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data,
                                     sampler=test_sampler,
                                     batch_size=args["test_batch_size"])

        test_loss = 0
        nb_test_steps = 0
        test_preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Prediction Iteration"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            tmp_test_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            test_loss += tmp_test_loss.mean().item()
            nb_test_steps += 1
            if len(test_preds) == 0:
                test_preds.append(logits.detach().cpu().numpy())
            else:
                test_preds[0] = np.append(test_preds[0], logits.detach().cpu().numpy(), axis=0)

        test_loss = test_loss / nb_test_steps
        test_preds = test_preds[0]
        test_preds = np.argmax(test_preds, axis=1)

        test_result = {}
        test_result["micro"] = f1_score(y_true=test_label_ids.numpy(),
                                        y_pred=test_preds, average="micro")

        test_result["test_loss"] = test_loss
        test_result["global_step"] = global_step
        test_result["loss"] = loss

        with open("../reports/new_bert_test_results.txt", "w") as fp:
            logger.info("******** Test results ********")
            for key in sorted(test_result.keys()):
                logger.info(" %s = %s", key, str(test_result[key]))
                fp.write("%s = %s\n" % (key, str(test_result[key])))
