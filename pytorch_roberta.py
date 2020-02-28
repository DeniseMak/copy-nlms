import pandas as pd
import argparse
import numpy as np
import json
import re
from tqdm import tqdm_notebook
from uuid import uuid4

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from pytorch-transformers import RobertaModel, RobertaTokenizer
from pytorch-transformers import RobertaForSequenceClassification, RobertaConfig
from pytorch-transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from pytorch-transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification

tokenizer = None
MAX_LEN = None
CUDA = torch.cuda.is_available()
if CUDA:
    print('Using Cuda')

class Data(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path).reset_index()
        self.len = len(self.data)
        
    def __getitem__(self, index):
        sentence = self.data.sent[index]
        label = self.data.label[index]
        X = prepare_features(sentence)
        y = torch.tensor(int(self.data.label[index]))
        return X, y

    def __len__(self):
        return self.len

def main():
    global MAX_LEN
    args = parse_all_args()
    
    model = get_model(args.model)

    MAX_LEN = get_seq_len(args.train)

    train_set, train_tmp = load_data(args.train, args.mb)
    test_set,  test_tmp = load_data(args.test, args.mb)

    model, train_preds, test_preds = train(args.lr, train_set, test_set, args.epochs, args.v, model)
    train_preds.to_csv(args.data.replace(".txt", "_train_preds.csv"))
    test_preds.to_csv(args.data.replace(".txt", "_test_preds.csv"))
    get_class('two hundred hundred', model, tokenizer)
    
def get_seq_len(path):
    """
    Get max sequence length for padding later
    """

    df = pd.read_csv(path)
    max_len = 0
    for row in df['sent']:
        toks = tokenizer.tokenize(row)
        curr_len = len(tokenizer.convert_tokens_to_ids(toks))
        if curr_len > max_len:
            max_len = curr_len
    # Account for additional tokens
    return max_len + 2

def get_model(model_name):
    """
    Load the model and tokenizer function specified by the user

    :param model_name: Name of the model
    :return model: Pretrained model
    """
    global tokenizer
    # NOTE: Do we need to use config??
    model = None
    if model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        config = RobertaConfig.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification(config)

    elif model_name == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        config = XLMConfig.from_pretrained('xlm-mlm-100-1280')
        model = XLMForSequenceClassification(config)

    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        config = BertConfig.from_pretrained('bert-base-multilingual-cased')
        model = BertForSequenceClassification.from_pretrained(config)

    config.num_labels = 2

    return model

def train(lr, train, test, epochs, verbosity, model):
    """
    Train a model using the specified parameters

    :param lr: Learning rate for the optimizer
    :param train: Training DataLoader
    :param test: Testing DataLoader
    :param verbosity: How often to calculate and print test accuracy
    :return model: trained model
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    train_preds = list()
    test_preds = list()

    # Check if Cuda is Available
    if CUDA:
        device = torch.device("cuda")
        model = model.cuda()

    model = model.train()

    for epoch in range(0, epochs):
        i = 0
        for x, y in train:
            optimizer.zero_grad()
            x = x.squeeze(1)

            output = model.forward(x)

            _, predicted = torch.max(output[0].detach(), 1)

            loss = loss_function(output[0], y)
            loss.backward()
            optimizer.step()
            if i % verbosity == 0:
                test_acc, preds = validation(model, test)
                print('({}.{:03d}) Loss: {} Test Acc: {}'.format(epoch, i, loss.item(), test_acc))
            i += 1
        train_acc, train_preds = validation(model, train)
        test_acc, test_preds = validation(model, test)
        print('({}.{:03d}) Loss: {} Train Acc: {} Test Acc: {}'.format(epoch, i, loss.item(), train_acc, test_acc))

    return model, pd.DataFrame(train_preds), pd.DataFrame(test_preds)

def validation(model, data):
    """
    Calculate accuracy of the model on a set of data

    :param model: Trained model
    :param data: Data set to predict and calculate on
    :return accuracy: Model accuracy on dataset
    """
    correct = 0
    total = 0

    predictions = torch.LongTensor()
    Y = torch.LongTensor()

    for x, y in data:

        x = x.squeeze(1)

        if CUDA:
            x = x.cuda()
            y = y.cuda()

        output = model(x)
        _, predicted = torch.max(output[0].detach(), 1)
        predictions = torch.cat((predictions, predicted))
        Y = torch.cat((Y, y))
        correct += (predicted.cpu() == y.cpu()).sum()
        total += x.shape[0]

    accuracy = correct.numpy() / total

    return accuracy, predictions

def read_file(path):
    """
    Get the lines of a specified file
    """
    f = open(path, "r")
    data = f.readlines()
    f.close()
    return data

def load_data(path, batch_size):
    """
    Load data for model
    """
    dataset = Data(path)

    params = {'batch_size': batch_size,
            'shuffle': True,
            'drop_last': False,
            'num_workers': 8}

    data_loader = DataLoader(dataset, **params)

    return data_loader, dataset

def prepare_features(seq):
    global MAX_LEN
    
    # Tokenzine Input
    tokens_a = tokenizer.tokenize(seq)

    # Truncate
    if len(tokens_a) > MAX_LEN - 2:
        tokens_a = tokens_a[0:(MAX_LEN - 2)]
    # Initialize Tokens
    tokens = [tokenizer.cls_token]
    # tokens.append()

    # Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    tokens.append(tokenizer.sep_token)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Zero-pad sequence length
    while len(input_ids) < MAX_LEN:
        input_ids.append(0)

    return torch.tensor(input_ids).unsqueeze(0)

def get_class(num, model, tokenizer):
    """
    Get class of a number in text form for an already trained model

    :param num: (str) Number in text form
    :return prediction: Class of input number
    """
    model.eval()
    num, _ = prepare_features(num)
    if CUDA:
        num = num.cuda()
    output = model(num)[0]
    _, pred_label = torch.max(output.data, 1)

    # For some reason 0, 1 maps to 1, 2
    prediction = pred_label + 1
    return prediction

def parse_all_args():
    """
    Parse args to be used as hyperparameters for model

    :return args: (argparse) Model hyperparameters 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-train",type=str,  help = "Path to input data file", \
        default = "./data/en_syn_sentences_train.txt")
    parser.add_argument('-test', help = 'Path to test data file', \
        type=str, default="./data/en_syn_sentences_test.txt")
    parser.add_argument("-model",type=str,  help = "Model type to use", default = "xlm")
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.01]",default=0.01)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    parser.add_argument("-v",type=int,\
            help="How often to calculate and print accuracy [default: 1]",default=1)
    parser.add_argument("-mb",type=int,\
            help="Minibatch size [default: 32]",default=32)

    return parser.parse_args()

if __name__ == '__main__':
    main()