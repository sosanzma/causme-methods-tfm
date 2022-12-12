"""
Created on Fri Jul 22 16:04:49 2022

@author: Manel Soler 

LSTM for causal discover
"""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import networkx as nx


# %% window generator

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None,
               label_columns=None,
               batch_size = None):
    # Store the raw data.
    self.train_df = train_df
    
    self.batch_size = batch_size

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
        self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}
    
    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])



# this function will split the data attending the window defined above

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


# we need a time series format foor feed the lstm
def make_dataset(self, data,batch_size ):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=batch_size,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df, self.batch_size)




WindowGenerator.train = train




# %% main function 

def lstm_cause(train_df,
               sens : float = 0.08,
               dropout : float = 0.1,
               maxlags: int = 1,
               batch_size : int = 16 ,
               lstm_neurons = None,
               epochs : int = 200,
               loss : str = "mse",
               graph : bool = False,
               patience : int = 8,
               noise : float = 0.05) -> np.ndarray: 
    
    '''
    Author : Manel Soler Sanz
    Granger causality with LSTM network for multi-dimensional time series
    Parameters:
    -----------
    data - input data (TxN)
    
    
    Returns:
    ----------
    coeff: coefficient matrix A. The ij-th entry in A represents the causal
    influence from j-th variable to the i-th variable.
    '''
    assert maxlags > 0
  
    T, N = train_df.shape
    train_df =pd.DataFrame(train_df)
    
    if lstm_neurons is None:
        lstm_neurons = int(N)
    print(lstm_neurons, maxlags)
    
  
    val_matrix = np.zeros((N, N))
    
    for j in range (N):   
             
        print(f"Calculating causality for serie {j} ... ")
        
        # build the network
        model = keras.Sequential()
        model.add(Dense(N,activation='sigmoid'))
        model.add(LSTM(lstm_neurons, return_sequences=False,
                        recurrent_activation="tanh", dropout=dropout))
        
        model.add(Dense(1,activation = 'linear'))
        
        
        # define our window
        window = WindowGenerator(train_df=train_df,batch_size = batch_size,
            input_width=maxlags, label_width=1, shift=1,label_columns = [j])
               
        
                
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                         patience=patience,
                                                          mode='min',
                                                          restore_best_weights=False)
        
        
        
        model.compile(loss=tf.losses.MeanSquaredError(),
                       optimizer=tf.optimizers.Adam(),
                       metrics=[tf.metrics.MeanAbsoluteError()])
        
        
        model.fit(window.train, epochs=epochs,verbose = 0, callbacks =[early_stopping])
        



   
        # now we compute the gradients for substract the importance of each variable in the forecasing
        grad = []
        for element in window.train:
            x = tf.constant(element[0])
            # we compute de gradient of y with  x
            with tf.GradientTape() as t:   
                t.watch(x)
                y = model(x)
            
            
            dy_dx = t.gradient(y, x)
            # mean of the absolute value of all the gradients of the 62 batch of 8 samples
            dy_dx = np.sum(np.abs(dy_dx.numpy()),axis=0)/len(dy_dx)
            # if the mean of all the batches is lower than noise value -> 0
            dy_dx[dy_dx < noise] =0  
            #sum of each lag contribution :
            dy_dx = np.sum(dy_dx,axis = 0) 
        
            grad.append(dy_dx)
        aux =  np.mean(grad,axis = 0)

        val_matrix[:,j] =  aux
    
        
    #val_matrix = normalize(val_matrix, norm='max', axis=0)
    # if the contribution of each gradient is lower than sens value -> 0      
    #val_matrix[val_matrix < sens] = 0 
    #print(lstm_model.summary())
    
    if graph :

        # we take a round value of the val_matrix  
        G = np.round(val_matrix-noise,1)
        G
        G =  np.array(G, dtype=bool)*1
        G = nx.from_numpy_matrix(G, create_using=nx.DiGraph)


        # Draw the graph using , we  want node labels.
        nx.draw(G,with_labels=True)
        plt.title(f"Graph causal relationships with {lstm_neurons} LSTM neurons")
        plt.show()
    
    return val_matrix

