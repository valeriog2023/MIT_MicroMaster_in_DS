#
# # THIS HOMEWORK IS TO PRACTICE WITH COLLABORATIVE FILTERING
#
# Given:
# 1) A (possibly very) sparse/incomplete matrix Y with dimension N x M where 
#      N represents the number of users
#      M represents a movie
#      the entry in the a,i position represents the score the user a gives to movie i
#      I use -1 to identify the absence of a value in Y
# 2) the hyperparameter lambda that determines how much importance is given to the normalization term
# 3) the hyper parameter K (rank) that determines the number of indpendent features used to parametrize the model
#
# we want to generate a matrix X in which we forecast a score for every combination of user/movie
# in particular we want to have values close to Y when Y is populated
#
# The way we proceed is to identify 2 vectors: U, V where 
#  U represens the users and has dimension N x K
#  V represnets the movies and has dimension K x M
#
# The algorithm we follow have these steps:
#  - we start with some randomly initialized vector V
#  - we compute and optimize an objective function to get the values of U
#  - we use the values of U identified to compute and optimize the same objective function in relation to V 
#  - we repeat the process again until U and V do not change anymore
#
# This let's us identify a minimum (local)
# The forecast matrix is given by U outer product V
import numpy as np
from typing import Callable


#
#  THIS MATRIX AND THE INIT VALUES ARE GIVEN
#
Y = np.array([ [ 5,-1, 7],
               [-1, 2,-1],
               [ 4,-1,-1],
               [-1, 3, 6]
            ])
#
# rank
K = 1
#
# lambda
L = 1
#
#
V = np.array([4,2,1])
U = np.array([6,0,3,6])


def forecast(U:np.array, V:np.array)->np.array:
    """
    this method returns the outer product  of the vectors U and V
    """
    return np.outer(U,V)



def empirical_risk(Y:np.array, U:np.array, V:np.array, L:float)->(float,float):
    """
    This method takes as input:
     - the original matrix Y (N x M) - we expect to use -1 where the value is missing here
     - U, V, vectors used to create the the forecast matrix: X
     - the value of lambda

    The function returns a tuple with 3 elements: 
      - the forecast matrix X
      - the squared error term, i.e.
           0.5 * Sum((Y_ai - X_ai)^2) 
        where Y_ai is defined

      - the regularization term, i.e.
         0.5 * L * (Norm(U)^2 + Norm(V)^2)
    """
    #
    # create the forecast matrix
    X = forecast(U,V)
    #
    # check the shape matches
    if Y.shape != X.shape:
        print("Matrices Y and X need to be of the same shape, instead we got:")
        print(Y)        
        print(X)                
        return -1,-1
    print("----------\nEmpirical Risk:")
    squared_terms = [ (a - b)**2 for a,b in zip(Y.flatten(),X.flatten()) if a != -1]
    squared_error = 0.5 * sum(squared_terms)
    #
    #
    normalization = 0.5 * L * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2)
    #
    #
    print(f"Squared error/Normalization: {squared_error} / {normalization}")
    return X,squared_error,normalization
    



def optimize_u_vector_step(Y:np.array, V:np.array, L:float)->np.array:
    """
    This function takes as input:
    Y: the data matrix given
    V: one of the 2 vectors used to generate the forecast matrix X (here it's considered constant)
    L: <float> set how important is the normalization element

    The function returns a new/optimized U array to use to generate the forecast matrix X
    """
    new_U = []
    for user_ratings in Y:
        new_U.append(optimize_single_u_step(user_ratings,V,L))
    #
    # 
    return np.array(new_U)    




def optimize_single_u_step(User_X_slice:np.array, V:np.array, L:float)->float:
    """
    This function takes as input:
    User_X_slice: an array which is a row from the data matrix that represents a single user set of scores
                  missing scores are assumed to be set to -1
    V: <np.array> this is one of the 2 vectors used to create the forecast matrix and it's considerd constant here
    L: <float> set how important is the normalization element

    The function returns a new value to be used for this user in the vector U
    The value is given by setting to zero the derivative of the empirical risk function keeping v constant              
    """
    #print(f"User_X_slice: {User_X_slice}")
    #print(f"User_X_forecast: {User_X_forecast}")    
    #print(f"Current V: {V}")
    num_elements = [ a * b for a,b in zip(User_X_slice,V) if a != -1 ]
    #print(f"Num components: {num_elements}")
    den_elements = [ b**2 for a,b in zip(User_X_slice,V) if a != -1 ]
    den_elements.append(L)
    #
    return round(sum(num_elements)/sum(den_elements),4)



if __name__ == "__main__":
    print(f"The Data Matrix is: \n{Y}")
    print(f"---\nUsing initial vectors U: {U} and V: {V} we have :")
    X,squared_error,normalization = empirical_risk(Y,U,V,L)
    print(f"Matrix forecast is: \n{X}")
    print("\n1) Optimize U vector:")
    new_U = optimize_u_vector_step(Y,V,L)
    print(f"New vector U is: {new_U}")

    