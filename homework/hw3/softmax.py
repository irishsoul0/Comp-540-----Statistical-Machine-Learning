import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  
  for i in xrange(m):

    p_i = np.dot(X[i, :], theta)
    p_i = p_i - np.max(p_i)
    p = np.exp(p_i) / np.sum(np.exp(p_i))
    J = J - np.log(p[y[i]])

    for j in xrange(theta.shape[1]):
      grad[:, j] = grad[:, j] + (p[j] - (j == y[i])) * X[i, :]

  J = J / m + reg * np.sum(theta ** 2)
  grad = grad / m 
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################

  K = theta.shape[1]
  
  h = (np.exp(np.dot(X, theta)).T / np.sum(np.exp(np.dot(X, theta)), axis = 1).T).T
  J = - 1.0 / m * np.sum(np.log(h)[np.arange(m), y]) + reg / (2 * m) * np.sum(np.square(theta))
    ## h is m * K
  exp_h = np.exp(np.dot(X, theta))
  
  
  one_zero = np.zeros((m, K))
  one_zero[np.arange(m), y] = 1
    
  grad = -1.0 / m * np.dot(X.T, one_zero - h) + reg / m * theta

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
