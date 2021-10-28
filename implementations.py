# Linear regression using gradient descent
from implementations import *
from helpers import * 

#Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    loss=0
    for n_iter in range(max_iters): 
        loss=compute_loss(y, tx, w)
        w= w - (gamma*compute_gradient(y, tx, w))
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=ws[0], w1=ws[1]))

    return losses, ws


#Linear regression using stochastic gradient descent
def least squares SGD(y, tx, initial w, max iters, gamma)



#Least squares regression using normal equations
def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w=np.linalg.solve(a, b)
    loss=compute_loss(y, tx, w)
    return loss, w



#Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w= np.linalg.solve(a, b)
    loss=compute_loss(y, tx, w)
    return loss, w


#Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    ws = [initial_w]
    losses = []
    w = initial_w
    loss=0

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return losses, ws



#Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    # init parameters
    ws = [initial_w]
    losses = []
    w = initial_w
    loss=0

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]


    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        ws.append(w)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break


            

            
            





















