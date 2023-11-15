#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

from tensorflow.keras.layers import Dense, Conv1D, LSTM, RNN

import warnings
warnings.filterwarnings('ignore')


# In[3]:


dhaka_model = keras.models.load_model("lstm_dhaka_model.h5")


# In[4]:


test_df = pd.read_csv('data/test.csv', index_col='datetime',parse_dates=True)


# In[5]:


def format_timeseries_data(df, input_length, output_length, target_names):
    
    if target_names is not None:
        target_indices = {name: i for i, name in enumerate(target_names)}
    col_indices = {name: i for i, name in enumerate(df.columns)}
    
    total_length = input_length + output_length
    
    input_slice = slice(0, input_length)
    output_slice = slice(input_length, None)
    
    data = np.array(df, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=total_length,
        sequence_stride=1,
        shuffle=False,
        batch_size=32
    )
    
    def split_to_input_output(x):
            
        inputs = x[:, input_slice, :]
        outputs = x[:, output_slice, :]
        
        if target_names is not None:
            outputs = tf.stack(
                [outputs[:,:,col_indices[name]] for name in target_names],
                axis=-1
            )

        inputs.set_shape([None, input_length, None])
        outputs.set_shape([None, output_length, None])
    
        return inputs, outputs
    
    ds = ds.map(split_to_input_output)
    
    return ds


# In[9]:


def predict(mo_lstm,data, days, steps):
    RD_max_train = 11.94 
    RD_min_train = 0.2
    RD_max_test =  28.11
    RD_min_test =  0.25

    R_max_train = 150.0
    R_min_train = 0.0
    R_max_test = 98.0
    R_min_test = 0.0

    test_ds_mo = format_timeseries_data(data, days, steps, ['river_discharge', 'rain_sum'])

    mo_sample_batch = next(iter(test_ds_mo))
    inputs, outputs = mo_sample_batch
    preds = mo_lstm(inputs)

    preds_array = preds[0]

    preds_np_array = np.array(preds_array)

    df_scaled = pd.DataFrame(preds_np_array)

    df = df_scaled.rename(columns={0: "river_discharge", 1: "rain_sum"})

    df['river_discharge'] = df['river_discharge'].apply(lambda x: x*(RD_max_train - RD_min_train) + RD_min_train)
    df['rain_sum'] = df['rain_sum'].apply(lambda x: x*(R_max_train - R_min_train) + R_min_train)
    
    return df


# In[11]:


pred_df = predict(dhaka_model,test_df,7,7)


# In[12]:


pred_df


# In[ ]:




