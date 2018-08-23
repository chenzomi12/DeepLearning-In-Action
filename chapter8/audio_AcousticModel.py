
# coding: utf-8

# In[9]:


import json
import numpy as np
import os
import matplotlib.pyplot as plt
import pprint as pp
import logging
import keras.backend as K
from keras.models import Model

from keras.layers import GRU, Input, TimeDistributed, BatchNormalization
from keras.layers import Bidirectional, Dense, Activation,Conv1D


# In[13]:


def gru_model(input_dim=161, output_dim=29, recur_layers=3, nodes=1024,
              kernel_size=11, conv_border_mode='valid',  conv_stride=2,
              initialization='glorot_uniform', batch_norm=True):
    """
    Building a recurrent neatual network (CTC) for speech
    with GRU units.
    """
    acoustic_input = Input(shape=(None, input_dim), name='the_input')
    
    conv_1d = Conv1D(nodes, kernel_size, padding=conv_border_mode,
                           strides=conv_stride, kernel_initializer=initialization,
                           activation='relu', name='conv1d')(acoustic_input)
    
    output = BatchNormalization(name='bn_conv1d')(conv_1d) if batch_norm else conv1d
    
    for i in range(recur_layers):
        output = GRU(nodes, activation='relu', kernel_initializer=initialization,
                    return_sequences=True, name='rnn_{}'.format(i+1))(output)
        
        if batch_norm:
            bn_layer = BatchNormalization(name='bn_rnn_{}'.format(i+1))
            otuput = bn_layer(output)
            
    y_pred = TimeDistributed(Dense(output_dim, activation='linear', 
                                  kernel_initializer=initialization, name='dense'))(output)
    
    model = Model(inputs=acoustic_input, outputs=y_pred)
    model.output_length = lambda x:x
    print(model.summary())
    return model
    
gru_model()

