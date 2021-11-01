# Linear regression using gradient descent
from computations import *


#Linear regression using gradient descent

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    
    for n_iter in range(max_iters): 
        gradient = compute_gradient(y, tx, w)
        w= w - (gamma*gradient)
        
    loss= compute_loss(y,tx,w) 

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
            gradient = compute_gradient(batch_y, batch_tx, w)
            loss = compute_loss(batch_y, batch_tx, w)
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


def logistic_regression(y, tx, initial_w, max_iters, gamma,threshold=1e-5):
    """Logistic regression using gradient descent for max_iters iteration given
    the input labelled data y, tx with initial_w and gamma as the initial weight and the
    learning rate respectively. Return final weights and loss. Uses stochastic gradient descent.""" 
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches = 1):
            # Compute gradient and update weight
            gradient = logistic_gradient (batch_y,batch_tx,w)
            w -= gamma*gradient
            # Compute loss
            loss = logistic_loss (batch_y,batch_tx,w)
            losses.append(loss)   
    return w, np.squeeze(losses[-1])
        

#Regularized logistic regression using gradient descent or SGD

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold = 1e-5): 
    """Regularized logistic regression using gradient descent.Returns final weights and loss"""
    
    w = initial_w
    losses = []
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=10000, num_batches = 1):
            loss = logistic_loss_reg(batch_y,batch_tx,w, lambda_)
            gradient = logistic_gradient_reg(batch_y,batch_tx,w, lambda_)
            w -= gamma*gradient
            
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, np.squeeze(losses[-1])
     