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

from pytorch_transformers import RobertaModel, RobertaTokenizer
from pytorch_transformers import RobertaForSequenceClassification, RobertaConfig

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
CUDA = torch.cuda.is_available()
if CUDA:
    print('Cuda is availible')

class Pairs(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        utterance = self.data.num[index]
        label = self.data.label[index]
        X, _ = prepare_features(utterance)
        y = torch.tensor(int(self.data.label[index]))
        return X, y

    def __len__(self):
        return self.len

def main():
    args = parse_all_args()
    train_set, test_set, label_to_ix = load_data(args.data, args.labels)

    print("Finished loading data")
    config = RobertaConfig.from_pretrained('roberta-base')
    config.num_labels = len(list(label_to_ix.values()))
    
    print("Beginning training")
    model = train(args.lr, train_set, test_set, args.epochs, args.v, config)
    get_class('two hundred hundred', model, label_to_ix)
    
def train(lr, train, test, epochs, verbosity, config):
    """
    Train a model using the specified parameters

    :param lr: Learning rate for the optimizer
    :param train: Training DataLoader
    :param test: Testing DataLoader
    :param verbosity: How often to calculate and print test accuracy
    :return model: trained model
    """
    model = RobertaForSequenceClassification(config)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

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
                print('({}.{}) Loss: {} Test Acc: {}'.format(epoch, i, loss.item(), test_acc))
        train_acc = validation(model, train)
        test_acc = validation(model, test)
        print('({}.{}) Loss: {} Train Acc: {} Test Acc: {}'.format(epoch, i, loss.item(), train_acc, test_acc))

    return model

def validation(model, data):
    """
    Calculate accuracy of the model on a set of data

    :param model: Trained model
    :param data: Data set to predict and calculate on
    :return accuracy: Model accuracy on dataset
    """
    correct = 0
    total = 0
    for sent, label in data:

        sent = sent.squeeze(0)
        if CUDA:
            sent = sent.cuda()
            label = label.cuda()
        output = model.forward(sent)[0]
        _, predicted = torch.max(output.data, 1)
        total += 1
        correct += (predicted.cpu() == label.cpu()).sum()
    accuracy = correct.numpy() / total

    return accuracy

def label_dict(dataset):
    """
    Map labels to indicies in a dictionary

    :param dataset: Data with labels
    :return label_to_ix: Label to index dictionary
    """
    label_to_ix = {}
    for label in dataset.label:
        for word in label.split():
            if word not in label_to_ix:
                label_to_ix[word] = len(label_to_ix)
    return label_to_ix

def read_file(path):
    """
    Get the lines of a specified file
    """
    f = open(path, "r")
    data = f.readlines()
    f.close()
    return data

def load_data(pair_path, label_path):
    """
    Load data for model
    """
    pairs = read_file(pair_path)
    labels = read_file(label_path)

    for i in range(0, len(pairs)):
        pairs[i] = pairs[i].split(", ")

    X = pd.DataFrame(pairs)
    y = pd.DataFrame(labels)
    dataset = pd.concat([X, y], axis=1, sort=False)
    dataset.columns = ['num', 'label']

    label_to_ix = label_dict(dataset)

    train_size = 0.8
    train_dataset = dataset.sample(
        frac=train_size, random_state=200).reset_index(drop=True)
    test_dataset = dataset.drop(train_dataset.index).reset_index(drop=True)

    print("FULL Dataset: {}".format(dataset.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = Pairs(train_dataset)
    testing_set = Pairs(test_dataset)

    params = {'batch_size': 1,
            'shuffle': True,
            'drop_last': False,
            'num_workers': 8}

    training_loader = DataLoader(training_set, **params)
    testing_loader = DataLoader(testing_set, **params)

    return training_loader, testing_loader, label_to_ix

def prepare_features(seq_1, max_seq_length=300, zero_pad=False, include_CLS_token=True, include_SEP_token=True):
    # Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

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

def get_class(num, model, label_to_ix):
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
    prediction = list(label_to_ix.keys())[pred_label]
    return prediction

def parse_all_args():
    """
    Parse args to be used as hyperparameters for model

    :return args: (argparse) Model hyperparameters 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-data",type=str,  help = "Path to input data file", default = "./data/en_syn_pair_words.txt")
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.01]",default=0.01)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    parser.add_argument('-labels', help = 'Path to input label file', \
            type=str, default="./data/en_syn_labels.txt")
    parser.add_argument("-v",type=int,\
            help="How often to calculate and print accuracy [default: 1]",default=1)

    return parser.parse_args()

if __name__ == '__main__':
    main()