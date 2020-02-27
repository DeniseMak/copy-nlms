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

from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification


tokenizer = None
CUDA = torch.cuda.is_available()
if CUDA:
    print('Cuda is availible')

class Pairs(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        utterance = self.data.sent[index]
        label = self.data.label[index]
        X, _ = prepare_features(utterance)
        y = torch.tensor(int(self.data.label[index]))
        return X, y

    def __len__(self):
        return self.len

def main():
    args = parse_all_args()
    
    train_set, train_tmp = load_data(args.train)
    test_set, test_tmp = load_data(args.test)

    print("Finished loading data")
    model = get_model(args.model)
    
    print("Beginning training")
    model, train_preds, test_preds = train(args.lr, train_set, test_set, args.epochs, args.v, model)
    train_preds.to_csv(args.data.replace(".txt", "_train_preds.csv"))
    test_preds.to_csv(args.data.replace(".txt", "_test_preds.csv"))
    get_class('two hundred hundred', model, tokenizer)
    
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
        print("Epoch: " + str(epoch))
        for i, (sent, label) in enumerate(train):
            optimizer.zero_grad()
            sent = sent.squeeze(0)
            if CUDA:
                sent = sent.cuda()
                label = label.cuda()
            output = model.forward(sent)[0]
            _, predicted = torch.max(output, 1)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            if i % verbosity == 0:
                test_acc = validation(model, test)
                print('({}.{}) Loss: {} Test Acc: {}'.format(epoch, i, loss.item(), test_acc[0]))
        train_acc, train_preds = validation(model, train)
        test_acc, test_preds = validation(model, test)
        print('({}.{}) Loss: {} Train Acc: {} Test Acc: {}'.format(epoch, i, loss.item(), train_acc, test_acc))

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
    predictions = list()
    for sent, label in data:

        sent = sent.squeeze(0)
        if CUDA:
            sent = sent.cuda()
            label = label.cuda()
        output = model.forward(sent)[0]
        _, predicted = torch.max(output.data, 1)
        predictions.append(predicted.item())
        total += 1
        correct += (predicted.cpu() == label.cpu()).sum()
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

def load_data(path):
    """
    Load data for model
    """

    dataset = pd.read_csv(path)
    dataset = Pairs(dataset)

    params = {'batch_size': 1,
            'shuffle': True,
            'drop_last': False,
            'num_workers': 8}

    data_loader = DataLoader(dataset, **params)

    return data_loader, dataset

def prepare_features(seq_1, max_seq_length=300, zero_pad=False, include_CLS_token=True, include_SEP_token=True):
    # Tokenzine Input
    tokens_a = tokenizer.tokenize(str(seq_1))

    # Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    # Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    # Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Input Mask
    input_mask = [1] * len(input_ids)
    # Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask

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

    parser.add_argument("-train",type=str,  help = "Path to input data file", default = "./data/en_syn_sentences_train.txt")
    parser.add_argument('-test', help = 'Path to test data file', \
        type=str, default="./data/en_syn_sentences_train.txt")
    parser.add_argument("-model",type=str,  help = "Model type to use", default = "xlm")

    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.01]",default=0.01)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    parser.add_argument("-v",type=int,\
            help="How often to calculate and print accuracy [default: 1]",default=1)

    return parser.parse_args()

if __name__ == '__main__':
    main()