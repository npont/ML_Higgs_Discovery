# Linear regression using gradient descent

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

def compute_stoch_gradient(y, tx, w, batch_size):
    l=0
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        l=compute_gradient(minibatch_y, minibatch_tx,w)
        
    return l


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

def sigmoid(t):
    return (1+np.exp(-t))**-1

def calculate_loss(y, tx, w):
    S=sigmoid(tx.dot(w))
    loss= y.T.dot(np.log(S))+ (1-y).T.dot(np.log(1-S))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def logistic_regression_loss(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

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
    
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    p=sigmoid(tx.dot(w))
    loss= calculate_loss(y, tx, w)+ (lambda_/2)*(la.norm(w)**2)
    grad=calculate_gradient(y, tx, w)+ 2*lambda_*w
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient= penalized_logistic_regression(y, tx, w, lambda_)
    w=w-gamma*gradient

    return loss, w



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




