from sklearn.model_selection import train_test_split
from num2words import num2words
import pandas as pd
import argparse
import random
import os
import re
import sys

def main():
    args = parse_all_args()
    pairs = list() # Integers in pairs to create ungrammatical numbers
    labels = list()

    pairs,labels = gen_pairs(args.range, args.samples) # Generate pairs of numbers
    text = to_text(pairs, args.lang) # Turn numbers into text
    text, labels = ungram_split(text, labels, args.samples, args.lang) # Make ungrammatical

    nums = list()
    for num in text:
        nums.append(' '.join(num))
    num_data = pd.DataFrame({'nums' : nums, 'labels' : labels})

    text = to_sent(text, args.lang, args.sent)
    all_data = pd.DataFrame({'sents' : text, 'labels' : labels})

    train_data, test_data = train_test_split(all_data, test_size=0.2)

    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    del train_data['index']
    del test_data['index']

    # Output
    train_data.to_csv(args.dir + args.lang + "_syn_train.csv")
    test_data.to_csv(args.dir + args.lang + "_syn_test.csv")
    num_data.to_csv(args.dir + args.lang + "_syn_nums.csv")

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

def gen_pairs(r, s):
    """
    - Generate random pairs of integers
    - Half of pairs have format [n, number > n by factor of 10] (or vice versa) (used to make ungrammatical number in text)
    - Half of pairs have format [n, -1] (used to make grammatical number in text)
    
    :param s: (int) Number of pairs (s/2 grammatical, s/2 ungrammatical)
    :param r: (int) Range of numbers to make pairs in [0,r]
    :return pairs: (list) Grammatical and ungrammatical pairs
    :return labels: (list) Denotes which pairs are ungrammatical (1) / grammatical (0)
    """

    pairs = list()
    labels = list()

    # Generate grammatical pairs
    for i in range(0, int(s/2)):
        p_i = []
        p_i.append(random.randint(2, r))
        p_i.append(-1)
        pairs.append(p_i)
        labels.append(0)

    # Generate ungrammatical pairs
    for i in range(0, int(s/2)):
        p_i = []

        # Generate a random number in range
        p_i.append(random.randint(2, r))

        num_str = str(p_i[0])
        j = len(num_str) - 1
        while num_str[j] == '0':
            j -= 1
        # j -= 1
        # if j != len(num_str) - 1:
        new = num_str[j:]
        new = ['0']*len(new)
        new[0] = '1'
        
        new = int(''.join(new))
        # print(int(new))
        rand = random.randint(new, r)
        p_i.append(rand)

        # Find that number's first digit to factor of 10
        # Ex. 125 -> 1000
        # fac10 = int("".join(str(p_i[0]) + "0"))
        
        # if fac10 >= r:
        #     # Make the number to add to pair smaller (still in range)
        #     num = int((p_i[0] / 10)) - 10
        #     rand = random.randint(0, num)
        #     p_i.append(rand)
        # else:
        #     # Make the number to add to pair bigger (still in range)
        #     rand = random.randint(fac10, r)
        #     p_i.append(rand)
        
        pairs.append(p_i)
        labels.append(1)

    return pairs, labels

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
            new[i] = new[i].replace('-', ' ')
            new[i] = new[i].replace(',', ' ')
            new[i] = re.sub(' +', ' ', new[i])
            new[i] = new[i].strip()
        text.append(new)

    return text

def to_sent(data, lang, sent):
    """
    Format number pairs write to a specified file
    :param path: (str) Filepath to write to
    :param data: (list) Data to format and write
    """

    # Get template sentences
    with open('./templates/' + lang + '_templates.txt', 'r', encoding="utf-8") as f:
        sentences = f.readlines()

    if not sent:
        sentences = ['***']
    sents = list()
    for item in data:
        string = random.choice(sentences).strip()
        if len(item) == 1:
            string = string.replace("***", item[0])
        else:
            string = string.replace("***", item[0] + " " + item[1])
        if lang == 'ja':
            string = string.replace(' ', '')
        sents.append(string)

    return sents

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
    parser.add_argument('-sent', dest='sent', help='Whether or not to generate numbers in sentences', action='store_true', default=False)

    args = parser.parse_args()

    if args.dir[-1] != "/":
        args.dir += "/"

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    if args.sent:
        args.dir += 'sent/'
    else:
        args.dir += 'no-sent/'

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)


    return args

if __name__ == '__main__':
    main()