# svm_mnist

In this project I implemented a multiclass SVM using linear and polynomial
kernels to make predictions on a handwritten digits dataset. I also implemented
hypothesis and instance transfer learning using cvxpy for quadratic
programming.

### About the dataset: The digits were written using a pen stylus by 40+
different authors.

The x,y position of the pen was captured eight times hence x1,y1 is the
position of the pen when it first touched the paper and x8,y8 is the last pen
position.  For ease of processing the co-ordinate system is 0-100 for both the
x and y dimensions.

Providing all 8 co-ordinate pairs makes the problem very easy so instead we
will use positions (x3,y3) to (x6,y6). 

### Installation instructions
Need python3.6+, sklearn, pandas, cvxpy

### Run instructions (optimal parameters)
`$ python multiclass_svm.py --n_splits 5 --num_steps 20000 --reg 8 --kernel linear`

`$ python multiclass_svm.py --n_splits 5 --num_steps 20000 --reg 16 --kernel poly`

`$ python transfer_svm.py`

### Results
The linear multiclass svm gets 29% error, the polynomial kernal multiclass svm
gets about 21% error, and both the hypothesis and instance transfer learners
improve on the simply binary classifier svm by about 5%.
