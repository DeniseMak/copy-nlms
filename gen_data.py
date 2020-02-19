from kanji_to_romaji import kanji_to_romaji
from num2words import num2words
import argparse
# import bangla
import random
import os

#NOTE: GET RID OF - AND " "

def main(): 
    args = parse_all_args()
    pairs = list()
    
    if args.load:
        pairs = load_prev(args.load)
    else:
        pairs = gen_pairs(args.range, args.samples)
        output_pairs(args.dir + "int_pairs.txt", pairs, "colon")
    
    if args.task == "semantic" or args.task == "both":
        labels = gen_sem_labels(pairs)
        final = to_text(pairs, args.lang)
        output_pairs(args.dir + args.lang + "_sem_pair_words.txt", final, "colon")

    if args.task == "syntax" or args.task == "both":
        pairs, labels = filter_pairs(pairs, args.samples)
        text = to_text(pairs, args.lang)
        output_pairs(args.dir + "syn_pair_words.txt", text, "and")
        output_indvs(args.dir + "syn_labels.txt", labels)

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
        line = line.split("; ")
        pair = [int(x) for x in line]
        pairs.append(pair)

    return pairs

def filter_pairs(pairs, s):
    """
    Generate random 'labels' for the syntactic task and use them to chose 
    whether to keep both numbers in a pair. Later, if there are 2 numbers in a
    pair they will be used to generate 'ungrammatical' words, and single 
    numbers will be grammatical

    :param pairs: (list) Randomly generated list of integer pairs
    :param s: (int) Number of pairs

    :return new_pairs: (list) Pairs with random entries reduced to length one
    :return labels: (list) Denotes which pairs have been reduced
    """
    labels = gen_ints(1, s)
    new_pairs = list()
    for i in range(0, s):
        new_pair = [pairs[i][0]]
        if labels[i] == 0:
            new_pair.append(-1)
        else:
            new_pair.append(pairs[i][1])
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
        if lang == 'ja':
            new[0] = kanji_to_romaji(new[0])
            if len(new) == 2:
                new[1] = kanji_to_romaji(new[1])
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

def output_pairs(path, data, mode):
    """
    Format data and write it to a specified file

    :param path: (str) Filepath to write to
    :param data: (list) Data to format and write
    :param mode: (str) How to join the data into strings
    """
    string = ""
    if mode == "colon":
        j = "; "
    # NOTE: What do do for ungrammatical data? I feel like the NN might just 
    # learn 'and' is more frequent in ungrammatical nums
    else:
        j = " and "
    for item in data:
        
        line = j.join(str(x) for x in item)
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
    parser.add_argument("-task",type=str,\
            help="Task for data to be generated", choices=['syntax','semantic','both'])
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