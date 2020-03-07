import editdistance
import argparse
import sys

def main():
    args = parse_all_args()
    distances_per_word = calc_distances(args.lang, args.window_size)
    opaqueness = calc_opaqueness(distances_per_word)
    with open(args.output, "a+") as f:
        f.write(args.lang + "," + str(args.window_size) + "," + str(opaqueness) + "\n")
    
def calc_distances(lang, window_size):
    """
    For each number word from 0 to 10,000, calculate a sum of pairwise levenshtein distances
    between it and window_size amount of previous number words.

    :param lang: (str) Language to calculate for
    :param window_size: (int) Given a word, how many previous words to calculate levenshtein distance with. Add each result to sum for that word.
    :return window_distances: (dict) A dictionary keyed on words with values corresponding to sums of levenshtein distances within window size.  
    """
    with open("data/numbers_" + lang + ".txt", "r", encoding="utf-8") as nums:
        window = []
        window_distances = {}

        for curr_word in nums:
            curr_word = curr_word.strip()

            # For when we are starting at zero
            if len(window) == 0:
                window.append(curr_word)
                window_distances[curr_word] = 0
            else:
                window_distance = 0 # Sum of levenshtein distances
                for window_word in window:
                    window_distance += editdistance.eval(curr_word, window_word)
                window_distances[curr_word] = window_distance

                # Remove first element from window if it's full
                if len(window) == window_size:
                    window.pop(0)

                # Insert the curr_word just processed
                window.append(curr_word)
        
        return window_distances

def calc_opaqueness(distances_per_word):
    """
    Function that calculates our opaqueness measure. Opaqueness is average of distances_per_word.

    param distances_per_word: (dict) A dictionary keyed on words with values corresponding to sums of levenshtein distances within window size.   
    """
    sum_distances = 0
    total = len(distances_per_word)
    for key in distances_per_word:
        sum_distances += distances_per_word[key]

    return sum_distances/total

def parse_all_args():
    """
    Parse args to be used as hyperparameters for model

    :return args: (argparse) Model hyperparameters 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-lang",type=str, help = "Language to get opaqueness measure", default = "en")
    parser.add_argument("-window_size",type=int, help= "Window size, window of previous words", default = 3)
    parser.add_argument("-output", type=str, help= "Where to output opaqueness results", default = "opaqueness_measures/results.csv")
    return parser.parse_args()

if __name__ == '__main__':
    main()