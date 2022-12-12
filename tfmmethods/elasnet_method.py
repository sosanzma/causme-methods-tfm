# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:14:40 2022

@author: 34695
"""
from sklearn.linear_model import ElasticNetCV
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

 

def var_elasnet(data: np.ndarray,
                    maxlags: int = 2,
                    cv: int = 5, 
                    random_state : int = 0,
                    max_iter : int = 10000,
                    tol = 1e-2,
                    graph : bool = False,
                    significance : int = 1) -> np.ndarray:
    '''
    Author : Manel Soler Sanz
    Granger causality with Elastic net regression for multi-dimensional time series
    Parameters:
    -----------
    data - input data (TxN)
    
    
    Returns:
    ----------
    coeff: coefficient matrix A. The ij-th entry in A represents the causal
    influence from j-th variable to the i-th variable.
    '''
        # check maxlags
    assert maxlags > 0        

    T, N = data.shape

    # stack data to form one-vs-all regression
    Y = data[maxlags:]

    # delayed time series
    X = np.hstack([data[maxlags - k:-k] for k in range(1, maxlags + 1)])

    elasnet_cv  = ElasticNetCV(cv=cv, random_state=0)

    # Matrix for coefficients
    coeff = np.zeros((N, N * maxlags))

    # Take one variable after the other as target
    for i in range(N):
      # calculate elasnet for heach variable target with the whole delayed series
        elasnet_cv.fit(X, np.ravel(Y[:, i])) 

        # Store coeff
        coeff[i] = np.abs(elasnet_cv.coef_)
    coeff = coeff.reshape(N, -1, N)    
    val_matrix = coeff.sum(axis = 1).T

    if graph :

        # we take as many deciamals as the value of significance to calculate the graph 
        
        # the higher "significance",the less permissive  the algorithm will be
        # and he more non-existent relationships will appear
        G =  np.round(val_matrix, significance)
        G = nx.from_numpy_matrix(G, create_using=nx.DiGraph)


        # Draw the graph using , we  want node labels.
        nx.draw(G,with_labels=True)
        plt.title(" Graph with significative causal relationships")
        plt.show()


    return val_matrix

 
