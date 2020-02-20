from kanji_to_romaji import kanji_to_romaji
from num2words import num2words
import argparse
import random
import os
import re

def main():     
     args = parse_all_args()
     pairs = list()
     
     if args.load:
          pairs = load_prev(args.load)
     else:
          pairs = gen_pairs(args.range, args.samples)
          output_pairs(args.dir + "int_pairs.txt", pairs)

     pairs, labels = filter_pairs(pairs, args.samples)
     
     text = to_text(pairs, args.lang)
     text, labels = ungram_split(text, labels, args.samples, args.lang)
     output_pairs(args.dir + args.lang + "_syn_pair_words.txt", text)
     output_indvs(args.dir + args.lang + "_syn_labels.txt", labels)

def ungram_split(pairs, labels, samples, lang):
    """
    Create the 'split' style ungrammatical words, atm only works 
    for english
    """
    
    loops = 0
    while labels.count(1) < samples // 2:
        # Prevent infinite loop if no points meet conditions
        if loops > 3:
            break
        loops += 1
        i = random.randint(0, samples - 1)
        if labels[i] == 0:
            if lang == 'en':
                pairs[i] = pairs[i][0].split(" and ")
            else:
                pairs[i] = pairs[i][0].split(" ")
            if len(pairs[i][0]) > 1:
                loops = 0
                pairs[i].append(pairs[i].pop(0))
                if lang == 'en':
                    joiner = random.choice([" and ", " "])
                else:
                    joiner = " "
                pairs[i] = [joiner.join(pairs[i])]
                labels[i] = 1

    return pairs, labels



def load_prev(path):
    """
    Load random integers from previously generated dataset

    :param path: (str) Path to integer pair file
    :return pairs: (list) Data from file as list of integer pairs
    """
    with open(path, "r") as f:
        lines = f.readlines()

    pairs = list()
    for line in lines:
        line = line.split(", ")
        pair = [int(x) for x in line]
        pairs.append(pair)

    return pairs

def filter_pairs(pairs, s):
    """
    Check whether the second number in a pair is larger than the first, by a factor of 10,
    in order to decide which numbers will be ungrammatical. Later, if there are 2 numbers in a
    pair they will be used to generate 'ungrammatical' words, and single 
    numbers will be grammatical

    :param pairs: (list) Randomly generated list of integer pairs
    :param s: (int) Number of pairs

    :return new_pairs: (list) Pairs with random entries reduced to length one
    :return labels: (list) Denotes which pairs have been reduced
    """
    labels = list()
    new_pairs = list()
    for i in range(0, s):
        new_pair = [pairs[i][0]]
        num_dig = len(str(new_pair[0]))
        for_check = ["0"] * num_dig
        for_check[0] = str(new_pair[0])[0]
        for_check = int("".join(for_check))

        # Add some randomization to the party
        if pairs[i][1] > for_check and random.choice([True, False]):
            new_pair.append(pairs[i][1])
            labels.append(1)
        else:
            new_pair.append(-1)
            labels.append(0)
        new_pairs.append(new_pair)

    return new_pairs, labels

def to_text(pairs, lang):
    """
    Convert positive integers in list of lists to word form

    :param pairs: (list) Integers to convert
    :param lang: (str) Language to convert them to

    :return text: (list) Integer pairs in word form
    """
    text = list()
    for pair in pairs:
        new = [num2words(pair[0], lang=lang)]
        if pair[1] > -1:
            new.append(num2words(pair[1], lang=lang))
        
        for i in range(0, len(new)):
            if lang == 'ja':
                new[i] = kanji_to_romaji(new[i])
            
            new[i] = re.sub('[^a-zA-Z0-9\n\.]', ' ', new[i])
            new[i] = re.sub(' +', ' ', new[i])
            new[i] = new[i].strip()
        text.append(new)

    return text

def gen_sem_labels(pairs):
    """
    Create labels based on whether pairs of integers are greater than (0), less 
    than (1), or equal (2) to each other.

    :param pairs: (list) Integer pairs to label
    :return labels: (list) Classes for given pairs
    """
    labels = list()
    for pair in pairs:
        if pair[0] > pair[1]:
            labels.append(0)
        elif pair[0] < pair[1]:
            labels.append(1)
        else:
            labels.append(2)
    return labels

def gen_pairs(r, s):
    """
    Create of list of [s] pairs of integers in range (0, r)

    :param r: (int) Max value for integers in list
    :param s: (int) Number of pairs to 
    :return pairs: (list) List of generated number pairs
    """
    p1 = gen_ints(r, s)
    p2 = gen_ints(r, s)
    pairs = [[p1[i], p2[i]] for i in range(0, s)] 
    return pairs

def output_pairs(path, data):
    """
    Format data and write it to a specified file

    :param path: (str) Filepath to write to
    :param data: (list) Data to format and write
    :param mode: (str) How to join the data into strings
    """
    string = ""
    
    for item in data:
        
        joiner = random.choice([" and ", " "])
        line = joiner.join(str(x) for x in item)
        string += line + "\n"

    with open(path, "w+") as f:
        f.write(string)

def output_indvs(path, data):
    """
    Format data and write it to a specified file

    :param path: (str) Filepath to write to
    :param data: (list) Data to format and write
    """
    string = ""
    for item in data:

        line = str(item)
        string += line + "\n"

    with open(path, "w+") as f:
        f.write(string)

def gen_ints(r, samples):
    """
    Generate a specified number of random integers

    :param r: (int) Max value of integers to be generated
    :param samples: (int) Amount of integers to be generated

    :return (list): Random integers in range (0,r) of length s
    """
    ints = list()
    for i in range(0, samples):
        ints.append(random.randint(0, r))
    return ints

def parse_all_args():
    """
    Parse commandline arguments and create folder for output if necessary

    :return args: Parsed arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-range",type=int,\
            help="Max value of integers to be generated [default=1000]",default=1000)
    parser.add_argument("-samples",type=int,\
            help="The number of integer pairs to be generated [default=100]",default=100)
    parser.add_argument("-dir",type=str,\
            help="Output directory for number pairs generated [default=data]", default="data")
    parser.add_argument("-lang",type=str,\
            help="Language of data to be generated [default=en]", default="en")
    parser.add_argument("-load",type=str,\
            help="Location of  previous set of integer pairs to create data")
    
    args = parser.parse_args()

    if args.dir[-1] != "/":
        args.dir += "/"

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    return args

if __name__ == '__main__':
    main()