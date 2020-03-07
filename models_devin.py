import argparse
import glob
import json
import os
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

class Data(Dataset):
    def __init__(self, sents, X, Y):
        self.sents = sents
        self.X = X
        self.Y = Y
        self.len = len(self.sents)
        
    def __getitem__(self, index):
        return self.sents[index], self.X[index], self.Y[index]

    def __len__(self):
        return self.len

MODEL_CLASSES = {
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

# Tasks and their labels
TASKS = {"syn" : [0, 1], "sem" : [0,1,2]}

def train(args, train_dataset, model, tokenizer, device):
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    model = model.to(device)
    model.train()

    for epoch in range(0, args.num_epochs):
        i = 0
        for sents, x, y in train_dataloader:
            # os.system("nvidia-smi")
            
            optimizer.zero_grad()

            x = x.squeeze(1)
            x = x.to(device)
            y = y.to(device)
            
            output = model(x, labels=y)
            loss = output[0]
            logits = output[1]

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits.detach(), 1)

            # Accuracy
            if i % args.verbosity == 0:
                correct = (predicted == y).float().sum()
                print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch ,i, loss.item(), correct/x.shape[0]))
            i += 1

            del x
            del y
            torch.cuda.empty_cache()

        # Get accuracy for epoch
        # my_print(out_f, '({}.{:03d}) Loss: {} Train Acc: {} Test Acc: {}'.format(epoch, i, loss.item(), train_acc, test_acc))

    return model


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_data(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_data(args, tokenizer, evaluate=False):

    # Open as pandas csv    
    if evaluate:
        data = pd.read_csv(args.data_dir + "/" + args.lang + "_" + args.task_name + "_" + "test.csv").reset_index()
    else:
        data = pd.read_csv(args.data_dir + "/" + args.lang + "_" + args.task_name + "_" + "train.csv").reset_index()

    # Get max sequence length of data
    max_len = 0
    for row in data['sents']:
        toks = tokenizer.tokenize(row)
        curr_len = len(tokenizer.convert_tokens_to_ids(toks))
        if curr_len > max_len:
            max_len = curr_len
    max_len += 2

    # Turn each sentence into vector x, add to X
    X = []
    for sentence in data['sents']:
        seq = sentence.split(';') # Works for both syn and sem tasks
        tokens = [tokenizer.cls_token]
        for s in seq:
            tokens_a = tokenizer.tokenize(s)
            for token in tokens_a:
                tokens.append(token)
            tokens.append(tokenizer.sep_token)
        x = tokenizer.convert_tokens_to_ids(tokens) # Vector representation of sentence
        while len(x) < max_len: # Zero-pad sequence length
            x.append(0)
        
        # Append sentence vector to total examples X
        X.append(x)
    
    # Convert to tensor
    X = torch.tensor(X).unsqueeze(0)

    # Get Y
    Y = data['labels']
    Y = torch.tensor(Y).unsqueeze(0)

    dataset = Data(data['sents'], X, Y)

    return dataset


def parse_args():
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="The input data dir. Should contain the .csv files for the task.",
    )
    parser.add_argument(
        "--model_type",
        default="distilbert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_string",
        default="distilbert-base-multilingual-cased",
        type=str,
        help="Path to pre-trained model or name"
    )
    parser.add_argument(
        "--task_name",
        default="syn",
        type=str,
        help="The name of the task to train on"
    )
    parser.add_argument(
        "--output_dir",
        default="./results/distilbert_multi",
        type=str,
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--lang", default = "en", type=str, help = "Language to run the task on")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity of training" )
    
    return parser.parse_args()


def main():

    # Get arguments
    args = parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare task
    if args.task_name not in TASKS:
        raise ValueError("Task not found: %s" % (args.task_name))
    label_list = TASKS[args.task_name]
    num_labels = len(label_list)

    # Get classes
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Set config
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_string,
        num_labels=num_labels,
        finetuning_task=args.task_name
    )
    
    # Set tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_string
    )

    # Set model
    model = model_class.from_pretrained(
        args.model_string,
        config=config
    )
    model = model.to(device)

    # Training
    train_dataset = load_data(args, tokenizer, evaluate=False)
    model = train(args, train_dataset, model, tokenizer, device)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    # # Save training arguments together with the trained model
    # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # # Load a trained model and vocabulary that you have fine-tuned
    # model = model_class.from_pretrained(args.output_dir)
    # tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    # model.to(args.device)

    # # Evaluation
    # results = {}
    # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    # checkpoints = [args.output_dir]
    # if args.eval_all_checkpoints:
    #     checkpoints = list(
    #         os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    #     )
    #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    # logger.info("Evaluate the following checkpoints: %s", checkpoints)
    # for checkpoint in checkpoints:
    #         global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #         prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=prefix)
    #         result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    #         results.update(result)

    # return results


if __name__ == "__main__":
    main()