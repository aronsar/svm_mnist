import pandas as pd
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument(
    '--input_file',
    '--f', 
    type=str, 
    default='./studentspen-train.csv', 
    help="The filepath of the input file.") 
parser.add_argument(
    '--n_splits',
    '--s', 
    type=int, 
    default=4, 
    help="The number of folds in k-fold cross validation.") 
args = parser.parse_args()


if __name__ == "__main__":
    data_and_digits = pd.read_csv(args.input_file)
    data = data_and_digits.iloc[:770, :-1]
    digits = data_and_digits.iloc[:770, -1]

    kf = KFold(n_splits=args.n_splits)
    import pdb; pdb.set_trace()
    for train_idxs, val_idxs in kf.split(data):
        X_train, X_val = data.iloc[train_idxs], data.iloc[val_idxs]
        y_train, y_val = digits.iloc[train_idxs], digits.iloc[val_idxs]
        

