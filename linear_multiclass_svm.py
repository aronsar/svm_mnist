import pandas as pd
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import argparse 
import random
import numpy as np

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
parser.add_argument(
    '--num_steps',
    '--n', 
    type=int, 
    default=1000, 
    help="The number of steps to train in SGD.") 
parser.add_argument(
    '--reg',
    '--r', 
    type=float, 
    default=1, 
    help="The regularization parameter.") 
args = parser.parse_args()


def highest_loss_class(weights, x_i, y_i):
    loss_vec = np.matmul(weights, x_i) - np.dot(weights[y_i,:], x_i)
    loss_vec[y_i] -= 1
    return np.argmax(loss_vec)


def train_linear_multiclass_svm(X_train, y_train):
    weights = np.zeros((10, 8))
    for step in range(1, args.num_steps):
        lr = 1 / (args.reg * step) # learning rate decay
        i = random.randint(0, X_train.shape[0]-1) # sample random data point
        x_i = X_train.iloc[i,:] 
        y_i = y_train.iloc[i]
        j_star = highest_loss_class(weights, x_i, y_i)

        if j_star != y_i:
            d_i = np.zeros(weights.shape)
            d_i[j_star,:] = x_i
            d_i[y_i,:] = -x_i
            weights = weights - lr*args.reg*weights - lr*d_i
        else:
            weights = weights - lr*args.reg*weights

    return weights
    


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
        
        weights = train_linear_multiclass_svm(X_train, y_train)
        val_num_wrong += len((np.argmax(np.dot(weights, X_val.T), axis=0) - np.array(y_val)).nonzero()[0])
        train_num_wrong += len((np.argmax(np.dot(weights, X_train.T), axis=0) - np.array(y_train)).nonzero()[0])
    
    print("Validation error rate: {:.3f}".format(val_num_wrong/data.shape[0]))
    print("Training error rate: {:.3f}".format(train_num_wrong/(data.shape[0]*(args.n_splits - 1))))
