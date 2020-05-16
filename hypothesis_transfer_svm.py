import pandas as pd
from sklearn.model_selection import KFold
import argparse 
import random
import numpy as np
from svm import SVM
from collections import defaultdict

parser = argparse.ArgumentParser() 
parser.add_argument(
    '--input_file',
    '--f', 
    type=str, 
    default='./studentspen-train.csv', 
    help="The filepath of the input file.") 
parser.add_argument(
    '--n_splits',
    '--n', 
    type=int, 
    default=5, 
    help="The number of folds in k-fold cross validation.") 
parser.add_argument(
    '--kernel',
    '--k', 
    type=str, 
    default="linear", 
    help="The kernel used by the svms; options are linear and poly.") 
parser.add_argument(
    '--num_steps',
    '--u', 
    type=int, 
    default=10000, 
    help="The number of steps to train in SGD.") 
parser.add_argument(
    '--reg',
    '--r',
    type=float,
    default=1,
    help="The regularization parameter.")
parser.add_argument(
    '--source_problem',
    '--s', 
    type=str, 
    default="1v7", 
    help="The binary svm to train as the source problem.") 
parser.add_argument(
    '--target_problem',
    '--t', 
    type=str, 
    default="1v9", 
    help="The binary svm problem to transfer the source weights to.") 
args = parser.parse_args()


def extract_source_target(data_and_digits, source_problem, target_problem):
    source_digits = np.array([int(d) for d in source_problem.split("v")])
    target_digits = np.array([int(d) for d in target_problem.split("v")])
    
    dnd = np.array(data_and_digits)
    source = dnd[dnd[:,-1] == source_digits[0]]
    source = np.vstack((source, dnd[dnd[:,-1] == source_digits[1]]))

    target = dnd[dnd[:,-1] == target_digits[0]]
    target = np.vstack((target, dnd[dnd[:,-1] == target_digits[1]]))

    return pd.DataFrame(source), pd.DataFrame(target)


if __name__ == "__main__":
    data_and_digits = pd.read_csv(args.input_file)
    source, target = extract_source_target(data_and_digits, args.source_problem, args.target_problem)
    source_data, target_data = source.iloc[:, :-1], target.iloc[:, :-1]
    source_digits, target_digits, = target.iloc[:, -1], target.iloc[:, -1]

    num_wrong = defaultdict(lambda: defaultdict(int))
    
    # first we train the source classifier
    source_svm = SVM(args.kernel, args, num_classes=2)
    source_svm.train(source_data, source_digits)

    # then train all the target classifiers
    kf = KFold(n_splits=args.n_splits, shuffle=True)
    for train_idxs, val_idxs in kf.split(target_data):
        X_train, X_val = target_data.iloc[train_idxs], target_data.iloc[val_idxs]
        y_train, y_val = target_digits.iloc[train_idxs], target_digits.iloc[val_idxs]
        
        # no transfer learning is done here
        n_svm = SVM(args.kernel, args, num_classes=2)
        n_svm.train(X_train, y_train)
        num_wrong["no_transfer"]["train"] += len((n_svm.predict(X_train) - np.array(y_train)).nonzero()[0])
        num_wrong["no_transfer"]["val"] += len((n_svm.predict(X_val) - np.array(y_val)).nonzero()[0])


        # hypothesis transfer learning
        h_svm = SVM(args.kernel, args, num_classes=2)
        h_svm.transfer(source_svm, X_train, y_train, type_="hypothesis")
        num_wrong["hypothesis_transfer"]["train"] += len((h_svm.predict(X_train) - np.array(y_train)).nonzero()[0])
        num_wrong["hypothesis_transfer"]["val"] += len((h_svm.predict(X_val) - np.array(y_val)).nonzero()[0])
        

        # instance transfer learning
        #i_svm = SVM(num_classes=2, args.kernel, args)
        #i_svm.transfer(type_="instance", source_svm, X_train, y_train)
    
    for transfer_type in num_wrong.keys():
        print(transfer_type + " training error rate: {:.3f}".format(num_wrong[transfer_type]["train"]/(data.shape[0]*(args.n_splits - 1))))
        print(transfer_type + " validation error rate: {:.3f}".format(num_wrong[transfer_type]["val"]/target_data.shape[0]))
        
