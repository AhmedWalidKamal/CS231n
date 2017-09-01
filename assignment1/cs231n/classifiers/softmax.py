import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
      scores = X[i].dot(W) # (1 x C) row vector, un-normalized log scores
      scores -= np.max(scores) # Avoiding numeric instability
      scores = np.exp(scores) / np.sum(np.exp(scores)) # exponentiate and normalize
      loss += -np.log(scores[y[i]])

      # computing the gradient: dL/dW = (dL/df) * (df/dW) = (p[i] - (j==y[i])) * (X[i])
      for j in range(num_classes):
          dW[:, j] += (scores[j] - (j == y[i])) * X[i]

  # Normalize then Regularize
  loss /= num_train
  loss += np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores, axis = 1, keepdims = True) # Note: keepdims=True enables Broadcasting by preventing dimensions from collapsing.
  scores = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)

  # I have no idea what I'm doing, but I'm trying to calculate the vectorized grad.
  scores[np.arange(num_train), y] -= 1
  dW = X.T.dot(scores)
  scores[np.arange(num_train), y] += 1 # urmmmmm revert it i guess.

  scores = -np.log(scores)
  loss += np.sum(scores[np.arange(num_train), y])

  # Normalize, Regularize
  loss /= num_train
  loss += np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
