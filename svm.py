import numpy as np
import pandas as pd
import random
import cvxpy as cp

class SVM:
    def __init__(self, kernel, args, num_classes):
        self.kernel = kernel
        self.weights = None
        self.args = args
        self.num_classes = num_classes
    
    def _kernel_fn(self, X):
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
        X_train = self._kernel_fn(X_train)
        self.weights = np.zeros((self.num_classes, X_train.shape[1]))

        # SGD
        for step in range(1, self.args.num_steps):
            lr = 1 / (self.args.reg * step) # learning rate decay
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
                self.weights = self.weights - lr*self.args.reg*self.weights - lr*d_i
            else:
                self.weights = self.weights - lr*self.args.reg*self.weights
    
    def transfer(self, source_svm, X_train, y_train, type_="hypothesis"):
        if type_ == "hypothesis":
            X = np.array(X_train)
            y = np.array(y_train)
            w_source = source_svm.weights

            w_target = cp.Variable(w_source.shape)
            w_target.value = w_source
            b = cp.Variable()

            objective = cp.Minimize(cp.sum_squares(w_target))
            constraints = [cp.multiply(y, (X @ (w_source + w_target) + b)) >= 1]
            prob = cp.Problem(objective, constraints)
            prob.solve()

            self.weights = w_target.value

        elif type_ == "instance":
            pass

    def predict(self, X):
        X = self._kernel_fn(X)
        return np.argmax(np.dot(self.weights, X.T), axis=0)
