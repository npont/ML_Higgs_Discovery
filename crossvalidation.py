import numpy as np
from implementations import *
from computations import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    index_te = k_indices[k]
    index_tr = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    index_tr = index_tr.reshape(-1)
    x_te=x[index_te]
    y_te=y[index_te]
    tx_te= build_poly(x_te, degree)
    x_tr=x[index_tr]
    y_tr=y[index_tr]
    tx_tr= build_poly(x_tr, degree)
    w = ridge_regression(y_tr, tx_tr, lambda_)
    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
    
    return loss_tr, loss_te




def cross_validation_ridge(y, x, k_indices, k,degree,lambda_):
    """return loss CV on a degree of a polynome."""
    
    # get k'th subgroup in test, others in train: TODO
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0])==k)]
    tr_index = tr_index.reshape(-1)

    x_te = x[te_index]
    y_te = y[te_index]
    x_tr = x[tr_index]
    y_tr = y[tr_index]
    
    #build polynome with corresponding degree to the corresponding feature identified by id_
    #least squares on tx_tr
    x_train_pol = build_poly(x_tr[:,:19],degree)
    tx_tr = np.concatenate((x_train_pol,x_tr[:,19:]),axis = 1)
    x_test_pol = build_poly(x_te[:,:19],degree)
    tx_te = np.concatenate((x_test_pol,x_te[:,19:]),axis =1)
    
    w, loss_tr = ridge_regression(y_tr,tx_tr,lambda_)
    
    # calculate the loss for train and test data: TODO
    loss_tr = np.sqrt(2 * loss_tr)
    loss_te = np.sqrt(2 * compute_loss(y_te,tx_te,w))
    return w,loss_tr, loss_te


def cross_validation_logistic(y, x, k_indices, k, lambda_,gamma,initial_w, max_iters, reg = False, SGD = False):
    """randomly partitions the data into k folds groups to train and test data."""
    stoch = SGD
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0])==k)]
    tr_index = tr_index.reshape(-1)

    x_te = x[te_index]
    y_te = y[te_index]
    x_tr = x[tr_index]
    y_tr = y[tr_index]
    
    if reg :
        w,loss_tr = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
        loss_te = logistic_loss_reg(y_te,x_te,w, lambda_)
    else :
        w, loss_tr = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
        loss_te = logistic_loss(y_te, x_te, w)
        
    return w,loss_tr, loss_te

