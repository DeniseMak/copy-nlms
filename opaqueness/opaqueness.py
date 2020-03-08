import argparse
import os
import sys

def main():
    args = parse_all_args()
    opaqueness = calc_opaqueness(args.lang)
    with open(args.output, "a+") as f:
        f.write(args.lang + "," + str(opaqueness) + "\n")
    
def calc_opaqueness(lang):
    """
    Function that calculates our opaqueness measure.
    """
    with open("./data/numbers_" + lang + ".txt", "r", encoding='utf-8') as f:
        seen = set()
        score = 0
        first = True
        for line in f:
            line = line.strip()
            if first:
                seen.add(line)
                first = False
                continue
            for j in range(0, len(line)):
                if line[0:j+1] in seen:
                    score += 1
            for j in range(len(line), -1, -1):
                if line[-j + 1:] in seen:
                    score += 1
            seen.add(line)
        return score - 1

def parse_all_args():
    """
    Parse args to be used as hyperparameters for model

    :return args: (argparse) Model hyperparameters 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-lang",type=str, help = "Language to get opaqueness measure", default = "en")
    parser.add_argument("-output", type=str, help= "Where to output opaqueness results", default = "results.csv")
    return parser.parse_args()

if __name__ == '__main__':
    main()