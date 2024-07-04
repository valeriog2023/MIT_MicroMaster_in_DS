import numpy as np
from typing import Callable
x = np.array([ [1,0,1],
               [1,1,1],
               [1,1,-1],
               [-1,1,1]
            ])
y = np.array([ 2, 2.7, -0.7, 2 ])

theta = np.array([0,1,2])

def hinge_loss(x:np.array ,y:float ,theta:np.array, epsiolon:float=0.000000001)-> float:
    """
    This function computes the hinge loss for a single datapoint given a specific theta
    Epsilon is use as a margine to iron out possible mathematical fluctuation

    The function computes the score:  y - theta.dot(x)
    and if the result is greater than 1 + epsilon, it return zero otherwise it returns 1 - score
    """
    score = y - theta.dot(x)
    if score >= 1 + epsiolon:
        return 0
    else:
        return 1 - score



def squared_error_loss(x:np.array ,y:float ,theta:np.array)-> float:
    """
    This function computes the squared error loss for a single datapoint given a specific theta

    The function returns: 0.5 * (y - theta.dot(x))^2
    
    """
    return (y - theta.dot(x))**2 /2

    
def emprical_risk(x:np.array,y:np.array,theta:np.array, loss:Callable[[np.array,np.array,float],float]) -> float:    
    """
    This function expects as argument:
    x: <np.array> the collection of data points, this is actually a matrix of size N x d where each row (N) is a datapoint of dimension (d)
    y: <np.array>  a one dimension array of size N that stores the observed value for each datapoint in x
    theta: <np.array> an array of size d, represents the regression vector
    loss: <Callable> it's a loss function that takes as input one datapoint, it's observed value, theta, and returns a float that represents a loss

    The function will return the value for the empirical risk as computed using the specific loss function
    (The empirical risk is just the mathematical average of the loss for each points)
    """
    N = x.shape[0]
    loss_vector = [ loss(x[i],y[i], theta) for i in range(N)]
    return sum(loss_vector)/N




if __name__ == '__main__':
    print(f"Computing the empirical risk using hinge loss: {emprical_risk(x,y,theta,hinge_loss)}")
    print(f"Computing the empirical risk using squared error loss: {emprical_risk(x,y,theta,squared_error_loss)}")    