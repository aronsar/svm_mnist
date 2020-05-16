import numpy as np
import pandas as pd
import random
import cvxpy as cp

class SVM:
    def __init__(self, kernel, args, classes):
        self.kernel = kernel
        self.weights = None
        self.args = args
        self.classes = classes
    
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
        self.weights = np.zeros((len(self.classes), X_train.shape[1]))

        # SGD
        for step in range(1, self.args.num_steps):
            lr = 1 / (self.args.reg * step) # learning rate decay
            i = random.randint(0, X_train.shape[0]-1) # sample random data point
            x_i = X_train.iloc[i,:]
            y_i = y_train.iloc[i]
            y_idx = self.classes.index(y_i)

            loss_vec = np.matmul(self.weights, x_i) - np.dot(self.weights[y_idx,:], x_i)
            loss_vec[y_idx] -= 1
            highest_loss_class_idx = np.argmax(loss_vec)

            if highest_loss_class_idx != y_idx:
                d_i = np.zeros(self.weights.shape)
                d_i[highest_loss_class_idx,:] = x_i
                d_i[y_idx,:] = -x_i
                self.weights = self.weights - lr*self.args.reg*self.weights - lr*d_i
            else:
                self.weights = self.weights - lr*self.args.reg*self.weights

        # save support vectors for instance transfer
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        w = self.weights[1,:] - self.weights[0,:]
        abs_margins = np.abs(np.matmul(X_train, np.expand_dims(w, axis=1)))
        support_vector_idxs = np.argpartition(abs_margins[:,0], 10)[:10]
        self.X_sv = X_train[support_vector_idxs, :]
        self.y_sv = y_train[support_vector_idxs]
    
    def hypothesis_transfer(self, source_svm, X_train, y_train):
        X = np.array(X_train)
        y = np.array(y_train)
        w_source = source_svm.weights.T
        
        # Aside: w_source is an 8x2 matrix, but it needs to be a hyperplane.
        # With some linear algebra we can convert it (and y) so that cvxpy
        # can solve the problem.
        
        # (w[:,1]*x > w[:,0]*x) == sign((w[:,1] - w[:,0])*x)
        w_source = w_source[:, 1] - w_source[:,0]

        # if the options for y are a and b, they now need to be -1 and 1
        y[y==1] = -1
        y[y==9] = 1
        
        C = 2
        w_target = cp.Variable(w_source.shape)
        slack = cp.Variable(X.shape[0])
        b = cp.Variable()

        objective = cp.Minimize(C * cp.sum_squares(w_target) + cp.sum_squares(slack) / X.shape[0])
        constraints = [cp.multiply(y, (X @ (w_source + w_target).T + b)) >= 1 - slack]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.weights = w_target.value + w_source
        self.b = b.value

    def instance_transfer(self, source_svm, X_train, y_train):
        X = np.array(X_train)
        y = np.array(y_train)
        X_sv = source_svm.X_sv
        y_sv = source_svm.y_sv

        # if the options for y are a and b, they now need to be -1 and 1
        y[y==1] = -1
        y[y==9] = 1
        y_sv[y_sv==1] = -1
        y_sv[y_sv==7] = 1

        C = 2
        w = cp.Variable(8)
        slack1 = cp.Variable(X_sv.shape[0])
        slack2 = cp.Variable(X.shape[0])
        b = cp.Variable()
        
        objective = cp.Minimize(C * cp.sum_squares(w) + cp.sum_squares(slack1) / X_sv.shape[0] + cp.sum_squares(slack2) / X.shape[0])
        constraints = [cp.multiply(y, (X @ w + b)) >= 1 - slack2, cp.multiply(y_sv, (X_sv @ w + b)) >= 1 - slack1]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        self.weights = w.value
        self.b = b.value

    def predict(self, X):
        X = self._kernel_fn(X)
        return np.argmax(np.dot(self.weights, X.T), axis=0)

    def predict_t(self, X):
        X = self._kernel_fn(X)
        return np.where(np.dot(self.weights, X.T) + self.b > 0, 9, 1)
