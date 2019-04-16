import os
from pprint import pprint
import numpy as np
import random
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc, f1_score

from data.definitions import DATA_PATH
from data.definitions import DATA_BERT, OUTPUT_BERT_DIR, PRETRAINED_BERT_CACHE

from bert.multilabel_data_processor import MultiLabelDataProcessor
from bert.cyclic_learning_rate import CyclicLR
from bert.input_examples_to_features import convert_examples_to_features
from bert.multilabel_bert_model import BertForMultiLabelSequenceClassification

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

model_state_dict = None


args = {
    "data_dir": DATA_BERT,
    "full_data_dir": DATA_PATH,
    "task_name": "sentiment_analysis",
    "no_cuda": True,
    "bert_model": "bert-base-uncased",
    "output_dir": OUTPUT_BERT_DIR,
    "max_seq_length": 128,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": 3e-5,
    "num_train_epochs": 4.0,
    "warmup_proportion": 0.1,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}


processors = {
    "sentiment_analysis": MultiLabelDataProcessor
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

    args["train_batch_size"] = int(args["train_batch_size"] / args["gradient_accumulation_steps"])

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args["seed"])

    task_name = args["task_name"].lower()
    if task_name not in processors:
        raise ValueError("Task {} not found!".format(task_name))

    processor = processors[task_name](args["data_dir"])
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(
        args["bert_model"], do_lower_case=args["do_lower_case"])

    train_examples = None
    num_train_steps = None

    if args["do_train"]:
        train_examples = processor.get_train_examples(args["full_data_dir"])
        num_train_steps = int(len(train_examples) / args["train_batch_size"] /
                              args["gradient_accumulation_steps"] * args["num_train_epochs"])

    # prepare model
    def get_model():
        if model_state_dict:
            model = BertForMultiLabelSequenceClassification.from_pretrained(
                args["bert_model"], num_labels=num_labels, state_dict=model_state_dict)
        else:
            model = BertForMultiLabelSequenceClassification.from_pretrained(
                args["bert_model"], num_labels=num_labels)
        return model

    model = get_model()

    if args["fp16"]:
        model.half()

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

    t_total = num_train_steps
    if args["local_rank"] != -1:
        t_total = t_total // torch.distributed.get_world_size()

    if args["fp16"]:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and \
                fp16 training.")

        optimizer = FusedAdam(optimizer_group_parameters,
                              lr=args["learning_rate"],
                              bias_correction=False,
                              max_grad_norm=1.0)

        if args["loss_scale"] == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args["loss_scale"])
    else:
        optimizer = BertAdam(optimizer_group_parameters,
                             lr=args["learning_rate"],
                             warmup=args["warmup_proportion"],
                             t_total=t_total)

    scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5,
                         step_size=2500, last_batch_iteration=0)

    eval_examples = processor.get_dev_examples(args["data_dir"])

    def eval():
        eval_features = convert_examples_to_features(
            eval_examples, args["max_seq_length"], tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                     batch_size=args["eval_batch_size"])

        all_logits = None
        all_labels = None

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)

                logits = model(input_ids, segment_ids, input_mask)

            tmp_eval_accuracy = accuracy_thresh(logits, label_ids)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(num_labels):
            fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'roc_auc': roc_auc}

        return result

    trian_features = convert_examples_to_features(
        train_examples, args["max_seq_length"], tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in trian_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in trian_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in trian_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in trian_features], dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if args["local_rank"] == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)

    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args["train_batch_size"])

    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh: float=0.5, sigmoid: bool=True):
        "Compute accuracy when `y_pred` and `y_true` are the same size."
        if sigmoid:
            y_pred = y_pred.sigmoid()
        return np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1).sum()

    def fit():
        num_epochs = args["num_train_epochs"]
        global_step = 0
        model.train()

        for i_ in range(int(num_epochs)):
            print("\nEpoch: {}".format(i_))

            tr_loss = 0
            nb_train_examples = 0
            nb_train_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean()
                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_train_examples += input_ids.size(0)
                nb_train_steps += 1

                # modify learning rate
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    lr_this_step = args["learning_rate"] * \
                        warmup_linear(global_step / t_total, args["warmup_proportion"])

                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            eval_results = eval()
            pprint(eval_results)

    model.unfreeze_bert_encoder()

    fit()

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(PRETRAINED_BERT_CACHE, "finetuned_pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    model_state_dict = torch.load(output_model_file)
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        args["bert_model"], num_labels=num_labels, state_dict=model_state_dict)
    model.to(device)

    eval_results = eval()
    with open("reports/bert_train_eval.txt", "w") as fp:
        pprint(eval_results, fp)

    def evaluate_model(model):
        predict_processor = MultiLabelDataProcessor(DATA_PATH)
        test_filename = "test.csv"
        test_examples = predict_processor.get_test_examples(DATA_PATH, test_filename)

        all_logits = None
        all_labels = None

        test_features = convert_examples_to_features(
            test_examples, args['max_seq_length'], tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.float)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data,
                                     sampler=test_sampler,
                                     batch_size=args["eval_batch_size"])
        model.eval()

        nb_eval_steps = 0
        nb_eval_examples = 0

        for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
            input_ids, input_mask, segment_ids, label_ids = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(num_labels):
            fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        f1 = {}
        for average in ["micro", "macro", "weighted"]:
            f1[average] = f1_score(all_labels.argmax(axis=1),
                                   all_logits.argmax(axis=1),
                                   average=average)

        result = {'roc_auc': roc_auc, 'f1': f1}
        return result

    final_results = evaluate_model(model)
    print('\n\nFinal Results:\n')
    pprint(final_results)
    with open("reports/bert_final_results.txt", "w") as fp:
        pprint(final_results, fp)
