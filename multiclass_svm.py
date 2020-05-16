import pandas as pd
from sklearn.model_selection import KFold
import argparse 
import numpy as np
from svm import SVM

parser = argparse.ArgumentParser() 
parser.add_argument(
    '--input_file',
    '--f', 
    type=str, 
    default='./studentspen-train.csv', 
    help="The filepath of the input file.") 
parser.add_argument(
    '--test_file',
    '--t', 
    type=str, 
    default='./studentspen-test.csv', 
    help="The filepath of the test file.") 
parser.add_argument(
    '--n_splits',
    '--s', 
    type=int, 
    default=5, 
    help="The number of folds in k-fold cross validation.") 
parser.add_argument(
    '--num_steps',
    '--n', 
    type=int, 
    default=20000, 
    help="The number of steps to train in SGD.") 
parser.add_argument(
    '--reg',
    '--r', 
    type=float, 
    default=8, 
    help="The regularization parameter.") 
parser.add_argument(
    '--kernel',
    '--k', 
    type=str, 
    default="linear", 
    help="The type of kernel to use; options are linear and poly.") 
args = parser.parse_args()


if __name__ == "__main__":
    data_and_digits = pd.read_csv(args.input_file)
    data = data_and_digits.iloc[:, :-1]
    digits = data_and_digits.iloc[:, -1]
    val_num_wrong = 0
    train_num_wrong = 0

    kf = KFold(n_splits=args.n_splits, shuffle=True)
    for train_idxs, val_idxs in kf.split(data):
        X_train, X_val = data.iloc[train_idxs], data.iloc[val_idxs]
        y_train, y_val = digits.iloc[train_idxs], digits.iloc[val_idxs]
        
        svm = SVM(args.kernel, args, classes=[0,1,2,3,4,5,6,7,8,9])
        svm.train(X_train, y_train)

        y_pred = svm.predict(X_train)
        train_num_wrong += len((y_pred - np.array(y_train)).nonzero()[0])
        y_pred = svm.predict(X_val)
        val_num_wrong += len((y_pred - np.array(y_val)).nonzero()[0])
    
    print("Validation error rate: {:.3f}".format(val_num_wrong/data.shape[0]))
    print("Training error rate: {:.3f}".format(train_num_wrong/(data.shape[0]*(args.n_splits - 1))))

    #data_and_digits = pd.read_csv(args.test_file)
