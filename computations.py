import numpy as np
from implementations import *



def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    loss = 1/2*np.mean(e**2)
    return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e)/len(e)
    return gradient


def logistic_loss(y,tx,w):
    """Compute the loss by negative log likelihood for the logistic regression."""
    pred = sigmoid(tx.dot(w)) 
    y = y[:,np.newaxis]
    return np.mean(np.log(1+np.exp(-np.multiply(y, pred))))


def logistic_loss_reg(y,tx,w, lambda_):
    """ Compute the loss by negative log likehlihood for the logistic regression with a
    penalizing term"""

    return logistic_loss(y,tx,w) + lambda_*np.squeeze(w.T.dot(w))

def logistic_gradient(y,tx,w):
    """ compute the gradient of logistic regression loss."""
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred-(y[:,np.newaxis]))
    return gradient

def logistic_gradient_reg(y, tx, w, lambda_):
    """Computes the gradient for the regularized L2 logistic gradient descent"""
    gradient = logistic_gradient(y, tx, w) + 2*lambda_*w
    return gradient
            
def sigmoid(t):
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w):
    S=sigmoid(tx.dot(w))
    loss= y.T.dot(np.log(S))+ (1-y).T.dot(np.log(1-S))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

# sampling

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)




def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset. Data can be randomly
    shuffled to avoid ordering in the original data messing with the
    randomness of the minibatches.
    
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
