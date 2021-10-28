import numpy as np
from implementations import *


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
    x_train_pol = polynomial_expansion(x_tr[:,:19],degree)
    tx_tr = np.concatenate((x_train_pol,x_tr[:,19:]),axis = 1)
    x_test_pol = polynomial_expansion(x_te[:,:19],degree)
    tx_te = np.concatenate((x_test_pol,x_te[:,19:]),axis =1)
    
    w, loss_tr = ridge_regression(y_tr,tx_tr,lambda_)
    
    # calculate the loss for train and test data: TODO
    loss_tr = np.sqrt(2 * loss_tr)
    loss_te = np.sqrt(2 * compute_loss(y_te,tx_te,w))
    return w,loss_tr, loss_te


def add_degrees(x_train, x_test,id_,degree):
    """adds x degrees to the feature identified by the index. """
    augmented_x_train =build_poly(x_train[:,id_],degree)
    
    x_train = np.delete(x_train,id_,axis=1) #delete old single feature
    x_train = np.concatenate((x_train,augmented_x_train),axis=1)
    
    augmented_x_test =build_poly(x_test[:,id_],degree)
    
    x_test = np.delete(x_test,id_,axis=1) #delete old single feature
    x_test = np.concatenate((x_test,augmented_x_test),axis=1)

    return x_train, x_test
