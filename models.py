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

from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

tokenizer = None
device = None
MAX_LEN = None
TASK = None

# if torch.cuda.is_available():
#     print('Using Cuda')
#     device = torch.device("cuda")
# else:
#     print('Using CPU')
#     device = torch.device("cpu")

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
        f.write(string)
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
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    elif model_name == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMForSequenceClassification.from_pretrained('xlm-mlm-100-1280')

    # elif model_name == 'bert':
    #     print('bert tokenzier')
    #     tokenizer = BertTokenizer.from_pretrained('/home2/lexilz/models/multi_cased_L-12_H-768_A-12/pytorch_model')
    #     print('yeet')
    #     model = BertForSequenceClassification.from_pretrained('./models/bert')
    #     print('rip')
    
    elif model_name == 'd-bert':
        # distilbert-base-multilingual-cased
        tokenizer = DistilBertTokenizer.from_pretrained('roberta-base')
        model = DistilBertForSequenceClassification.from_pretrained('roberta-base')


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
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    model = model.to(device)
    

    for epoch in range(0, epochs):
        model.train()
        i = 0
        open(out_f, "w+").close() # Clear out previous log files
        for sents, x, y in train:

            optimizer.zero_grad()
            
            loss, predicted, y = get_preds(x, y, model)

            loss.backward()
            optimizer.step()
        
            # print(evaluate_data(test, model))

            # Accuracy
            if i % verbosity == 0:
                # print
                correct = (predicted == y).float().sum()
                print("Epoch ({}.{}), Loss: {:.3f}, Accuracy: {:.3f}".format(epoch ,i, loss.item(), correct/x.shape[0]))
            i += 1
            break

        # Get accuracy for epoch
        # model.eval()
                    
        # train_res = evaluate_data(train, model)
        model.eval()

        test_path = './results/test_preds.csv'
        open(test_path, "w+").close() 
        evaluate_data(test, model, test_path)

        test_res = pd.read_csv(test_path)

        # train_res.to_csv(out_f.replace('.txt', 'train_preds.csv'))
        # test_res.to_csv(te

        # train_acc = len(np.where(train_res['preds'] == train_res['true'])) / len(train_res)
        test_acc = len(np.where(test_res['preds'] == test_res['true'])) / len(test_res)

        my_print(out_f, '({}.{:03d}) Loss: {} Train Acc: {} Test Acc: {}'.format(epoch, i, loss.item(), test_acc, test_acc))
        exit()

    return model

def evaluate_data(data, model, path):

    with open(path, 'a') as f:
        f.write('sents, true, preds\n')
    # all_sents = list()
    # all_preds = list()
    # all_true = list()
        for sents, x, y in data:
            sents = list(sents)
            predicted = get_preds(x, y, model).tolist()
            # all_preds += list(predicted)
            y = y.tolist()

            rows = list(zip(sents, y, predicted))
            for row in rows:
                f.write(', '.join([str(i) for i in row]) + '\n')


def get_preds(x, y, model):
    x = x.squeeze(1)
    x = x.to(device)
    y = y.to(device)
    
    output = model(x, labels=y)
    loss, logits = output

    _, predicted = torch.max(logits.detach(), 1)

    return loss, predicted, y

def validation(model, data):
    """
    Calculate accuracy of the model on a set of data

    :param model: Trained model
    :param data: Data set to predict and calculate on
    :return accuracy: Model accuracy on dataset
    """
    correct = 0
    total = 0
      
    predictions = (torch.LongTensor()).to(device)
    Y = (torch.LongTensor()).to(device)

    model = model.to(device)
    model.eval()

    for sents, x, y in data:

        x = x.squeeze(1)

        x = x.to(device)
        y = y.to(device)

        output = model(x)

        _, predicted = torch.max(output[0].detach(), 1)
        predicted = predicted.to(device)
        predictions = torch.cat((predictions, predicted))
        Y = torch.cat((Y, y))
        correct += (predicted.cpu() == y.cpu()).sum()
        total += x.shape[0]

        # del x
        # del y
        # del predicted
        # torch.cuda.empty_cache()

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
            'shuffle': False,
            'drop_last': False,
            'num_workers': 8}

    data_loader = DataLoader(dataset, **params)

    return data_loader, dataset

def prepare_features(seq):
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

def get_class(num, model, tokenizer):
    """
    Get class of a number in text form for an already trained model

    :param num: (str) Number in text form
    :return prediction: Class of input number
    """
    model.eval()
    num, _ = prepare_features(num)
    num = num.to(device)
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