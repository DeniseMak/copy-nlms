import pandas as pd
import argparse
import numpy as np
import json
import sys
import re
from tqdm import tqdm_notebook
from uuid import uuid4

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from transformers import XLMRobertaModel, XLMRobertaTokenizer
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig
from transformers import RobertaModel, RobertaTokenizer
from transformers import obertaForSequenceClassification, RobertaConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

tokenizer = None
device = None
MAX_LEN = None
TASK = None

class Data(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path).reset_index()
        self.len = len(self.data)
        
    def __getitem__(self, index):
        sentence = self.data.sents[index]
        label = self.data.labels[index]
        X = prepare_features(sentence)
        y = torch.tensor(int(label))
        return sentence, X, y

    def __len__(self):
        return self.len

def main():
    global MAX_LEN
    global TASK
    global device

    args = parse_all_args()
    open(args.out_f, "w+").close() # Clear out previous log files
    TASK = args.task
    
    if torch.cuda.is_available():
        my_print(args.out_f,'Using Cuda')
        device = torch.device("cuda")
    else:
        my_print(args.out_f,'Using CPU')
        device = torch.device("cpu")
 
    my_print(args.out_f, 'Loading model')
    model = get_model(args.model)
    my_print(args.out_f, 'Getting max sequence len')
    MAX_LEN = get_seq_len(args.train)

    my_print(args.out_f, 'Max seq len = {}\nLoading data...'.format(MAX_LEN))

    train_set, train_tmp = load_data(args.train, args.mb)
    test_set,  test_tmp = load_data(args.test, args.mb)
    my_print(args.out_f, 'Data loaded')

    my_print(args.out_f, "Starting training")
    model = train(args.lr, train_set, test_set, args.epochs, args.v, model, args.out_f)
    my_print(args.out_f, 'Finished training \n Outputting results')
    
def my_print(path, string, verbosity=True):
    with open(path, 'a+') as f:
        f.write(string + "\n")
    if verbosity:
        print(string)

def get_seq_len(path):
    """
    Get max sequence length for padding later
    """

    df = pd.read_csv(path)
    max_len = 0
    for row in df['sents']:
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
    model = None
    if model_name == 'roberta':
        tokenizer = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base')
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base')

    elif model_name == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMForSequenceClassification.from_pretrained('xlm-mlm-100-1280')

    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

    elif model_name == 'd-bert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')

    elif model_name == 'en-roberta':
        tokenizer = RobertaForSequenceClassification.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    return model

def train(lr, train, test, epochs, verbosity, model, out_f):
    """
    Train a model using the specified parameters

    :param lr: Learning rate for the optimizer
    :param train: Training DataLoader
    :param test: Testing DataLoader
    :param verbosity: How often to calculate and print test accuracy
    :return model: trained model
    """
    optimizer = optim.Adadelta(params=model.parameters(), lr=lr)

    model = model.to(device)
    
    for epoch in range(0, epochs):
        model.train()
        i = 0
        for sents, x, y in train:

            optimizer.zero_grad()
            
            loss, predicted, y = get_preds(x, y, model)

            loss.backward()
            optimizer.step()
        
            # Accuracy
            if i % verbosity == 0:
                correct = (predicted == y).float().sum()
                my_print(out_f, "Epoch ({}.{}), Loss: {:.3f}, Accuracy: {:.3f}".format(epoch ,i, loss.item(), correct/x.shape[0]))
            i += 1
            break

        # Get test accuracy for full data each epoch
        model.eval()

        test_path = out_f.replace('.txt', '_test_preds.csv')
        test_acc = evaluate_data(test, model, test_path) 

        my_print(out_f, '({}.{:03d}) Loss: {} Test Acc: {}'.format(epoch, i, loss.item(), test_acc))

    # Make train predictions at the end and get accuracy
    train_path = out_f.replace('.txt', '_train_preds.csv')
    train_acc = evaluate_data(train, model, train_path)    

    my_print(out_f, '({}.{:03d}) Loss: {} Train Acc: {}'.format(epoch, i, loss.item(), train_acc))

    return model

def evaluate_data(data, model, path):
    """
    Get and store predictions for a dataset in a specified csv file

    :param data: (DataLoader) Contains minibatches to get predictions on
    :param model: Model used to make predictions
    :param path: Location to store predictions

    :return acc: (float) Accuracy of the predictions made
    """
    open(path, "w+").close() # clear prev preds/create file
    with open(path, 'a', encoding='utf-8') as f:
        f.write('sents,true,preds\n')

        for sents, x, y in data:
            sents = list(sents)
            _, predicted, _ = get_preds(x, y, model)
            predicted = predicted.tolist()
            y = y.tolist()

            rows = list(zip(sents, y, predicted))
            for row in rows:
                f.write(','.join([str(i) for i in row]) + '\n')

    res = pd.read_csv(path)
    correct = np.where(res['preds'] == res['true'])
    acc = len(correct[0]) / len(res)

    return acc

def get_preds(x, y, model):
    """
    Get predictions from a model on a specified set of vectors

    :param x: Input vectors
    :param y: Correct labels for input x
    :param model: Model to evaluate on
    """
    x = x.squeeze(1)
    x = x.to(device)
    y = y.to(device)
    
    output = model(x, labels=y)
    loss, logits = output

    _, predicted = torch.max(logits.detach(), 1)

    return loss, predicted, y

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
    Create a DataSet and a DataLoader for the model

    :param path: (str) Location of data
    :param batch_size: (int) Size of minibatches
    """
    dataset = Data(path)

    params = {'batch_size': batch_size,
            'shuffle': True,
            'drop_last': False,
            'num_workers': 8}

    data_loader = DataLoader(dataset, **params)

    return data_loader, dataset

def prepare_features(seq):
    """
    Convert a sequence of one or more sentences into tokens, seperating sentences
    with the seperator token

    :param seq: (str) Text data to tokenize
    :return (tensor): Tokenized text data
    """
    global MAX_LEN

    seq = seq.split(';')
    tokens = list()
    # Initialize Tokens
    tokens = [tokenizer.cls_token]
    for s in seq:
        # Tokenzine Input
        tokens_a = tokenizer.tokenize(s)

        # Add Tokens and separators
        for token in tokens_a:
            tokens.append(token)

        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Zero-pad sequence length
    while len(input_ids) < MAX_LEN:
        input_ids.append(0)

    return torch.tensor(input_ids).unsqueeze(0)

def parse_all_args():
    """
    Parse args to be used as hyperparameters for model

    :return args: (argparse) Model hyperparameters 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-train",type=str,  help = "Path to input data file", \
        default = "./data/en_syn_train.csv")
    parser.add_argument("-task",type=str,  help = "Whether to to the syntactic or semantic task", \
        default = "syn")
    parser.add_argument('-test', help = 'Path to test data file', \
        type=str, default="./data/en_syn_test.csv")
    parser.add_argument("-out_f",type=str,  help = "Path to output acc file", \
        default = "./results/res")
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