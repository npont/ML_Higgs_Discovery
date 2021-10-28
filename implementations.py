# Linear regression using gradient descent
from computations import *


#Linear regression using gradient descent

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


#Linear regression using stochastic gradient descent

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
