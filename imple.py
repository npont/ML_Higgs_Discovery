# Linear regression using gradient descent

# computations
# Useful starting lines
import numpy as np


def compute_loss(y, tx, w):
    e = y - np.matmul(tx,w)
    loss = sum(e**2) / (2 * len(tx))
    return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.matmul(tx,w)
    gradient = - np.matmul(np.transpose(tx),e)/len(tx)
    loss = sum(e**2) / (2 * len(tx))
    return (loss, gradient)

def logistic_loss(y, tx, w):
    """  
    Calculate the negative loss likelihood.
    Parameters
    """
    N = y.shape[0]
    loss = 0
    for i in range(N):
        aux = np.dot(np.transpose(tx[i,:]),w)
        loss = loss - (y[i]*np.log(sigmoid(aux)) + (1-y[i])*np.log(1-sigmoid(aux)))
    return loss[0]

def logistic_gradient(y, tx, w):
    """
    Compute the gradient for negative log likelihood.
    Parameters
    """
    gradient = np.dot(np.transpose(tx), sigmoid(np.dot(tx,w)) - y)
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

            
            
            
            
            

#implementations


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters): 
        loss, gradient = compute_gradient(y, tx, w)
        w= w - (gamma*gradient)
        ws.append(w)
        losses.append(loss)

    return w, loss

    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.
    Parameters
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size = 100):
            loss, gradient = compute_gradient(batch_y, batch_tx, w)
            w = w - gradient * gamma
            ws.append(w)
            losses.append(loss)

    return w, loss

#Linear regression using stochastic gradient descent

#Least squares regression using normal equations

def least_squares(y, tx):
    # Compute weights
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    
    #Calculate loss
    e = y - np.matmul(tx, w)
    loss = sum(e*e) / (2*len(tx))
    
    return w, loss


#Ridge regression using normal equations

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(tx)
    aI = lambda_ * N * 2 * np.identity(len(tx[0]))
    a = np.transpose(tx).dot(tx) + aI
    b = np.transpose(tx).dot(y)
    w= np.linalg.solve(a, b)
    
    e = y - np.matmul(tx, w)
    loss = sum(e*e) / (2*len(tx))
    
    return w, loss

#Logistic regression using gradient descent or SGD


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get gradient and update w
        gradient = logistic_gradient(y,tx,w)
        w = w - gamma*gradient
    
    loss = logistic_loss(y,tx,w)
    
    return w, loss

     

#Regularized logistic regression using gradient descent or SGD

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    # init parameter
    w = initial_w


    # start the logistic regression
    for iter in range(max_iters):
        # get gradient and update w.
        gradient = logistic_gradient(y,tx,w) + lambda_*w 
        w = w - gamma*gradient
    
    loss = compute_loss_logistic(y,tx,w)
    
    return w, loss
