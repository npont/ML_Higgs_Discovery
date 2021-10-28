import numpy as np
from implementations import *



def compute_loss(y, tx, w):
    y=np.reshape(y, (-1, 1))
    e=y-np.dot(tx,w)
    mse=(1/(2*tx.shape[0]))*np.dot(e.transpose(),e)
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    y=np.reshape(y, (-1, 1))
    e=y-np.dot(tx,w)
    L=(-1/tx.shape[0])*np.dot(tx.transpose(), e)
    return L

def calculate_gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


#Linear regression using stochastic gradient descent
def compute_stoch_gradient(y, tx, w, batch_size):
    l=0
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        l=compute_gradient(minibatch_y, minibatch_tx,w)
        
    return l




def compute_logistic_gradient(y,tx,w):
    """ compute the gradient of logistic regression loss."""
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred-(y[:,np.newaxis]))
    return gradient


def compute_logistic_gradient_regularized(y, tx, w, lambda_):
    """Computes the gradient for the regularized L2 logistic gradient descent"""
    gradient = compute_logistic_gradient(y, tx, w) + 2*lambda_*w
    return gradient


