# TODO: Make batch sizes bigger than 1
# TODO: Check that output makes sense
# TODO: Make generic for all model
# TODO: Check that it works for each language
# TODO: Do we need the config file?

# Misc
import pandas as pd
import argparse
import numpy as np
import json
import re
from tqdm import tqdm_notebook
from uuid import uuid4

# Torch/Transformers
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification

# Globals
tokenizer = None
CUDA = torch.cuda.is_available()

class SynDataset(Dataset):
    """
    A wrapper for our data so that it is compatible with Pytorch.
    """
    def __init__(self, df):
        self.data = df

    def __getitem__(self, index):
        sentence = self.data.sent[index]
        label = self.data.label[index]
        X = prepare_features(sentence)
        y = torch.tensor(int(self.data.label[index]))
        return X, y

    def __len__(self):
        return len(self.data)

def main():
    # Parse arguments and get data
    args = parse_all_args()

    # Load in data with pandas
    all_data = pd.read_csv(args.data).drop_duplicates()
    train_data = all_data.sample(frac=args.percent_train,random_state=200).reset_index(drop=True)
    test_data = all_data.drop(train_data.index).reset_index(drop=True)

    # Format to use with pytorch
    params = {'batch_size': args.batch_size,
            'shuffle': True,
            'drop_last': False,
            'num_workers': 8}
    train_loader = DataLoader(SynDataset(train_data), **params)
    test_loader = DataLoader(SynDataset(test_data), **params)

    model = get_model(args.model)

    # Train and evaluate model on training/testing sets
    train_predictions, test_predictions = train(model, args.lr, train_loader, test_loader, args.epochs, args.v)
    
    # # Output testing and training predictions
    # train_predictions.to_csv(args.data.replace(".txt", "_train_preds.csv"))
    # test_predictions.to_csv(args.data.replace(".txt", "_test_preds.csv"))

    # # Classify a single example
    # classify_example('two hundred hundred', model, label_to_ix)
    
def train(model, lr, train_data, test_data, epochs, verbosity):
    """
    Train the linear classifier that is on top of the pretrained model.

    :param lr: Learning rate for the optimizer
    :param train: Training DataLoader
    :param test: Testing DataLoader
    :param verbosity: How often to calculate and print test accuracy
    :return model: trained model
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    train_predictions = list()
    test_predictions = list()

    # Check if Cuda is Available
    if CUDA:
        device = torch.device("cuda")
        model = model.cuda()

    # Set model into train mode
    model = model.train() 

    # Training loop
    for epoch in range(0, epochs):
        print("Epoch: " + str(epoch))
        for i, (x, y) in enumerate(train_data):
            optimizer.zero_grad() # Reset gradients for each batch
            x = x.squeeze(0)
            if CUDA:
                sent = sent.cuda()
                label = label.cuda()
            
            # Sent the example forward pass and get result
            output = model.forward(x)[0]

            # Get loss and make backward pass
            loss = loss_function(output, y)
            loss.backward()
            
            # Adjust the weights
            optimizer.step()

            # Print an incremental test accuracy result
            if i % verbosity == 0:
                test_acc = evaluate(model, test_data)
                print('({}.{}) Loss: {} Test Acc: {}'.format(epoch, i, loss.item(), test_acc[0]))

        # Get overall train/test accuracy (after training completely done)
        train_acc, train_predictions = evaluate(model, train)
        test_acc, test_predictions = evaluate(model, test)
        print('({}.{}) Loss: {} Train Acc: {} Test Acc: {}'.format(epoch, i, loss.item(), train_acc, test_acc))

    return pd.DataFrame(train_predictions), pd.DataFrame(test_predictions)

def evaluate(model, data):
    """
    Calculate accuracy of the model on classifying a set of data

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

def label_dict(dataset):
    """
    Map labels to indices in a dictionary

    :param dataset: Data with labels
    :return label_to_ix: Label to index dictionary
    """
    label_to_ix = {}
    for label in dataset.label:
        for word in label.split():
            if word not in label_to_ix:
                label_to_ix[word] = len(label_to_ix)
    return label_to_ix

def prepare_features(seq_1, max_seq_length=300, zero_pad=False, include_CLS_token=True, include_SEP_token=True):
    """
    Prepare sentences for being passed into model of choice (BERT, etc.)
    """
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

def classify_example(x, model, label_to_ix):
    """
    Get class of an example

    :param x: (str) Example data
    :return prediction: List of label probabilities
    """
    model.eval()
    x, _ = prepare_features(x)
    if CUDA:
        num = num.cuda()
    output = model(x)[0]
    _, pred_label = torch.max(output.data, 1)
    prediction = list(label_to_ix.keys())[pred_label]
    return prediction

def get_model(model_name):
    """
    Load the model and tokenizer function specified by the user

    :param model_name: Name of the model
    :return model: Pretrained model
    """
    global tokenizer
    model = None
    if model_name == 'roberta':
        print('Loading roberta')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification('roberta-base')

    elif model_name == 'xlm':
        print('Loading xlm')
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMForSequenceClassification('xlm-mlm-100-1280')

    elif model_name == 'bert':
        print('Loading bert')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

    model.config.num_labels = 2

    return model

def parse_all_args():
    """
    Parse args to be used as hyperparameters for model

    :return args: (argparse) Model hyperparameters 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-data",type=str,  help = "Path to input data file", default = "./data/en_syn_data.csv")
    parser.add_argument("-percent_train", type=float, help="Percent of data to use as training (float)", default = .80)
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.01]",default=0.01)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    parser.add_argument("-v",type=int,\
            help="How often to calculate and print accuracy [default: 1]",default=1)
    parser.add_argument("-batch_size", type=int, help="How large batches should be (int)", default=32)
    parser.add_argument("-model",type=str,  help = "Model type to use", default = "xlm")

    return parser.parse_args()

if __name__ == '__main__':
    main()
