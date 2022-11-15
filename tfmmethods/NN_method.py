"""Vector autoregressive models VAR.

Model that fit VAR models and use coefficients as scores
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout,  Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import normalize




#%% Neuran network architecture
def MLP_(
    data_shape ,
    Dense_neurons : list = [128,32,8] ,
    add_Dropout : bool = False,
    Dropout_rate : float = 0.2, 
    activation : str = "relu",
    o_activation : str = "linear",
    loss : str = "mse") -> keras.engine.functional.Functional  :
    """
    Parameters
    ----------
    Dense_layers : int
        Number of Dense layers after GRU layers.
    Dense_neurons : list
        List with the numbers of neurons in each fully-connecred layer.
    add_Dropout : bool
        Specifies whether dropout regularization should be applied.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
    activation : str 
        Activation function 
    o_activation : str
        Activation function in output layer    
    data_shape : tuple
        Shape of the training data.
    Returns
    -------
    model : keras.engine.training.Model
        Model with the specified architecture.
    """
    # data_shape[1] - lag, data_shape[2] - number of signals
    input_layer = Input((data_shape[1]))
    

    layers_dense = Dense(Dense_neurons[0], activation=activation)(input_layer)
    # Adding Dropout
    if add_Dropout:
        layers_dense = Dropout(Dropout_rate)(layers_dense)
    # Adding Dense layers
    for densel in range(1, len(Dense_neurons)):
        layers_dense = Dense(Dense_neurons[densel], activation=activation)(layers_dense)
        # Adding Dropout
        if add_Dropout:
            layers_dense = Dropout(Dropout_rate)(layers_dense)

    # Adding output layer
    output = Dense(1, activation=o_activation)(layers_dense)

    model = Model(inputs=input_layer, outputs=output)
    
    model.compile(loss=loss, optimizer="adam")

    return model

# %% main function 

def nn_cause2(data: np.ndarray,
               maxlags: int = 1,
               epochs : int = 150,
               sens : float = 0.1,
               Dense_neurons : list = [128,32,8] ,
               add_Dropout : bool = False,
               Dropout_rate : float = 0.0, 
               activation : str = "relu",
               o_activation : str = "linear",
               loss : str = "mse"
               ) -> np.ndarray :
        '''
        Granger causality with neural network for multi-dimensional time series
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
        
        res = np.zeros((N, N * maxlags))
        
        # Take one variable after the other as target
        for j in range(N):
             # calculate nn regression for each variable target with the whole delayed series
            model = MLP_(data_shape =X.shape, Dense_neurons = Dense_neurons,
                         add_Dropout= add_Dropout, Dropout_rate = Dropout_rate,
                         activation = activation, o_activation = o_activation,
                         loss = loss)
            # fit model 
            model.fit( X, Y[:,j], epochs=epochs, verbose=0)
            
            x = tf.constant(X)
            # we compute de gradient of y with  x
            with tf.GradientTape() as t:   
                t.watch(x)
                y = model(x)
            
            
            dy_dx = t.gradient(y, x)
            # mean of the absolute value of all the gradients from each series
            dy_dx = np.sum(np.abs(dy_dx.numpy()),axis=0)/len(dy_dx)
            
            
            #dy_dx = (dy_dx - dy_dx.min()) / (dy_dx.max() - dy_dx.min())
            # threshold : All values under `sens` value will be 0
            dy_dx[dy_dx < sens] =0
            
            # add it to the results matrix
            res[j,:] = dy_dx
    
        val_matrix =res.reshape(N, -1, N) 
    
        
        # sum the contribtion of each lag
        val_matrix= val_matrix.sum(axis = 1).T
        val_matrix = normalize(val_matrix, norm='max', axis=0)
        
        
            
        return (val_matrix)