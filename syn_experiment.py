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
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
CUDA = torch.cuda.is_available()
if CUDA:
    print('Cuda is availible')

class SynDataset(Dataset):
    """
    A wrapper for our data so that it is compatible with Pytorch.
    Takes in a pd dataframe and turns it into Pytorch compatible Dataset object.
    """
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        utterance = self.data.sent[index]
        label = self.data.label[index]
        X, _ = prepare_features(utterance)
        Y = torch.tensor(int(self.data.label[index]))
        return X, Y

    def __len__(self):
        return self.len

def main():
    # Parse arguments and get data
    args = parse_all_args()
    train_set, test_set, label_to_ix, = load_data(args.data, args.labels)

    # Train and evaluate model on training/testing sets
    model, train_predictions, test_predictions = train(args.lr, train_set, test_set, args.epochs, args.v)
    
    # Output testing and training predictions
    train_predictions.to_csv(args.data.replace(".txt", "_train_preds.csv"))
    test_predictions.to_csv(args.data.replace(".txt", "_test_preds.csv"))

    # Classify a single example
    classify_example('two hundred hundred', model, label_to_ix)
    
def train(lr, train_data, test_data, epochs, verbosity):
    """
    Train the linear classifier that is on top of the pretrained model.

    :param lr: Learning rate for the optimizer
    :param train: Training DataLoader
    :param test: Testing DataLoader
    :param verbosity: How often to calculate and print test accuracy
    :return model: trained model
    """
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
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
            optimizer.zero_grad() # Reset gradients for each example
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

    return model, pd.DataFrame(train_preds), pd.DataFrame(test_preds)

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

def load_data(sentences_path, label_path):
    """
    :param pair_path: (str) Path to pairs file
    :param label_path: (int) Path to labels file
    :return training_loader (DataLoader): Pytorch dataloader for training data
    :return testing_loader (DataLoader): Pytorch dataloader for testing data
    :return label_to_ix (dict): Assign an index to each label val
    """

    # Read in sentences
    sents = []
    with open(sentences_path, "r") as sentences:
        for sentence in sentences:
            sents.append(sentence.strip())
    X = pd.DataFrame(sents)

    # Read in labels
    labs = []
    with open(label_path, "r") as labels:
        for label in labels:
            labs.append(label.strip())
    Y = pd.DataFrame(labs)

    # Merge into one pandas table
    dataset = pd.concat([X, Y], axis=1, sort=False)
    dataset.columns = ['sent', 'label']

    # Assign an index to each possible label
    label_to_ix = label_dict(dataset)

    # Split the dataset based on train/test
    train_size = 0.8
    train_dataset = dataset.sample(
        frac=train_size, random_state=200).reset_index(drop=True)
    test_dataset = dataset.drop(train_dataset.index).reset_index(drop=True)
    print("FULL Dataset: {}".format(dataset.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Write train/test to CSV files
    train_dataset.to_csv(sentences_path.replace(".txt", "_train.txt"))
    test_dataset.to_csv(sentences_path.replace(".txt", "_test.txt"))

    # Parameters for pytorch data loader
    params = {'batch_size': 1,
            'shuffle': True,
            'drop_last': False,
            'num_workers': 8}

    # Make pytorch dataloaders, have to wrap train/test with pytorch dataset class
    train_dataset = SynDataset(train_dataset)
    test_dataset = SynDataset(test_dataset)
    training_loader = DataLoader(train_dataset, **params)
    testing_loader = DataLoader(test_dataset, **params)

    return training_loader, testing_loader, label_to_ix

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

def parse_all_args():
    """
    Parse args to be used as hyperparameters for model

    :return args: (argparse) Model hyperparameters 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-data",type=str,  help = "Path to input data file", default = "./data/en_syn_sentences.txt")
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
