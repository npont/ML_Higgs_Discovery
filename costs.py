import numpy as np


def sigmoid(t):
    return (1+np.exp(-t))**-1

def calculate_mse(e):
    """Calculate the mse for vector e"""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e"""
    return np.mean(np.abs(e))


def calculate_loss(y, tx, w):
    S=sigmoid(tx.dot(w))
    loss= y.T.dot(np.log(S))+ (1-y).T.dot(np.log(1-S))
    return np.squeeze(- loss)


def logistic_regression_loss(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
     
    ws = [initial_w]
    losses = []
    w = initial_w
    loss=0
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss=compute_loss(minibatch_y, minibatch_tx, w)
            w= w - (gamma*compute_stoch_gradient(minibatch_y, minibatch_tx, w,batch_size))
            ws.append(w)
            losses.append(loss)
        
    return losses, ws




    
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    p=sigmoid(tx.dot(w))
    loss= calculate_loss(y, tx, w)+ (lambda_/2)*(la.norm(w)**2)
    grad=calculate_gradient(y, tx, w)+ 2*lambda_*w
    return loss, grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w



def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient= penalized_logistic_regression(y, tx, w, lambda_)
    w=w-gamma*gradient
    return loss, w