import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    contributers = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        contributers += 1
        dW[:, j] += X[i]
    dW[:, y[i]] -= X[i] * contributers

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and gradient.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Idea: 1) subtract from each row in scores the value of scores[i][y[i]]
  # 2) Maximize all elements between 0 and themselves
  # 3) Sum all elements and add the sum to the loss, then normalize loss.

  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y]
  correct_class_scores = correct_class_scores.reshape((num_train, 1))
  scores += 1 # Adding delta = 1
  scores -= correct_class_scores # Broadcasting the column vector to perform (1)
  scores[np.arange(num_train), y] = 0 # Disabling correct class scores from loss.
  scores = np.maximum(scores, 0) # contributers in each row = num of non zero numbers.
  loss += np.sum(scores)
  loss /= num_train
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  binary_scores = scores
  binary_scores[scores > 0] = 1 # Elements = 1 contribute to loss, 0 doens't contribute
  row_sum = np.sum(binary_scores, axis = 1) # row vector, each element is the # of contributers to this class across all examples
  binary_scores[np.arange(num_train), y] = -row_sum.T # Put # of contributers of each class in the coresspoding places for the y(i)s
  dW = X.T.dot(binary_scores)

  # Normalize
  dW /= num_train

  # Regularize
  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
