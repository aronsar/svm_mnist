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
parser.add_argument(
    '--kernel',
    '--k', 
    type=str, 
    default="linear", 
    help="The type of kernel to use; options are linear and poly.") 
args = parser.parse_args()

NUM_CLASSES = 10

class SVM:
    def __init__(self, kernel):
        self.kernel = kernel
        self.weights = None
    
    def kernel_fn_(self, X):
        if self.kernel == "linear":
            return X
        elif self.kernel == "poly":
            X = np.array(X)
            X = np.hstack((np.ones((X.shape[0],1)), X)) # add a column of ones at the left
            il = np.tril_indices(X.shape[1]) # indices of lower diagonal
            out_X = np.zeros((X.shape[0], len(il[0])))
            X = np.expand_dims(X, axis=2)
            for i, row in enumerate(X):
                mult = np.matmul(row, row.T)
                out_X[i,:] = mult[il]

        return pd.DataFrame(out_X)

    def train(self, X_train, y_train):
        X_train = self.kernel_fn_(X_train)
        self.weights = np.zeros((NUM_CLASSES, X_train.shape[1]))

        for step in range(1, args.num_steps):
            lr = 1 / (args.reg * step) # learning rate decay
            i = random.randint(0, X_train.shape[0]-1) # sample random data point
            x_i = X_train.iloc[i,:]
            y_i = y_train.iloc[i]

            loss_vec = np.matmul(self.weights, x_i) - np.dot(self.weights[y_i,:], x_i)
            loss_vec[y_i] -= 1
            highest_loss_class = np.argmax(loss_vec)

            if highest_loss_class != y_i:
                d_i = np.zeros(self.weights.shape)
                d_i[highest_loss_class,:] = x_i
                d_i[y_i,:] = -x_i
                self.weights = self.weights - lr*args.reg*self.weights - lr*d_i
            else:
                self.weights = self.weights - lr*args.reg*self.weights

    def predict(self, X):
        X = self.kernel_fn_(X)
        return np.argmax(np.dot(self.weights, X.T), axis=0)

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
        
        svm = SVM(kernel=args.kernel)
        svm.train(X_train, y_train)

        y_pred = svm.predict(X_train)
        train_num_wrong += len((y_pred - np.array(y_train)).nonzero()[0])
        y_pred = svm.predict(X_val)
        val_num_wrong += len((y_pred - np.array(y_val)).nonzero()[0])
    
    print("Validation error rate: {:.3f}".format(val_num_wrong/data.shape[0]))
    print("Training error rate: {:.3f}".format(train_num_wrong/(data.shape[0]*(args.n_splits - 1))))
