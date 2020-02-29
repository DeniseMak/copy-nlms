import editdistance
import argparse

def main():
    args = parse_all_args()
    calc_distances(args.lang, args.window_size)

def calc_distances(lang, window_size):
    with open("numbers_" + lang, "r") as nums:
        window = []
        distances = []
        for line in nums:
            if len(window) == 0:
                distances.append(0)
                window.append(line)
            else:
                for item in window:
                    distances.append(editdistance(line, item))
                    

            

                

def parse_all_args():
    """
    Parse args to be used as hyperparameters for model

    :return args: (argparse) Model hyperparameters 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-lang",type=str, help = "Language to get opaqueness measure", default = "en")
    parser.add_argument("-window_size",type=int, help= "Window size, window of previous words", default = 3)
    return parser.parse_args()

if __name__ == '__main__':
    main()