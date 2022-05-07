# Libraries to install
import argparse
from tqdm import tqdm
import utils

# Parse the arg
arg_parse = argparse.ArgumentParser()
arg_parse.add_argument('--eval_corpus_path', help="Path for eval", default=None)
arguments = arg_parse.parse_args()

# Make a list to store predictions
preds = [] 

for line in tqdm(open(arguments.eval_corpus_path)):
    # Append London to predictions
    preds.append('London')

# Evaluation
max, true = utils.evaluate_places(arguments.eval_corpus_path, preds)

# Printing out results
print('Results that are correct: {} out of {}: {}%'.format(true, max, true / max * 100))