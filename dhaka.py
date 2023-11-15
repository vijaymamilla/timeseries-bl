#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Model, Sequential

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

from tensorflow.keras.layers import Dense, Conv1D, LSTM, RNN

import warnings
warnings.filterwarnings('ignore')


# In[32]:


plt.rcParams["figure.figsize"] = (9,6)


# In[33]:


np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)


# In[34]:


df_raw = pd.read_csv('data/dhaka_raw.csv',index_col=['datetime'],parse_dates=True)


# In[35]:


df_raw.head()


# In[36]:


df_raw.info()


# In[37]:


df_raw.isnull().sum()


# In[38]:


df_raw['sealevelpressure']


# In[39]:


df = df_raw.fillna(method='ffill')


# In[40]:


df.isnull().sum()


# In[41]:


fig, ax = plt.subplots(figsize=(13,6))

ax.plot(df['rain_sum'])
ax.set_xlabel('Date')
ax.set_ylabel('Rain')

fig.autofmt_xdate()
plt.tight_layout()


# In[42]:


fig, ax = plt.subplots(figsize=(13,6))

ax.plot(df['river_discharge'])
ax.set_xlabel('Date')
ax.set_ylabel('River Discharge')

fig.autofmt_xdate()
plt.tight_layout()


# In[43]:


df.describe().transpose()


# In[44]:


n = len(df)

# Split 70:20:10 (train:validation:test)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

train_df.shape, val_df.shape, test_df.shape


# In[45]:


RD_max_train = np.max(train_df['river_discharge'])
RD_min_train = np.min(train_df['river_discharge'])
RD_max_test = np.max(test_df['river_discharge'])
RD_min_test = np.min(test_df['river_discharge'])

R_max_train = np.max(train_df['rain_sum'])
R_min_train = np.min(train_df['rain_sum'])
R_max_test = np.max(test_df['rain_sum'])
R_min_test = np.min(test_df['rain_sum'])


# In[46]:


print(RD_max_train,RD_min_train,RD_max_test,RD_min_test)

print(R_max_train, R_min_train,R_max_test,R_min_test)


# In[47]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_df)
test_scaler = MinMaxScaler()
test_scaler.fit(test_df)

train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
val_df[val_df.columns] = scaler.transform(val_df[val_df.columns])
test_df[test_df.columns] = test_scaler.transform(test_df[test_df.columns])


# In[48]:


train_df.head()


# In[49]:


train_df.describe().transpose()


# In[50]:


#train_df.to_csv('data/train.csv')
#val_df.to_csv('data/val.csv')
#test_df.to_csv('data/test.csv')


# In[51]:


train_df = pd.read_csv('data/train.csv', index_col='datetime',parse_dates=True)
val_df = pd.read_csv('data/val.csv', index_col='datetime',parse_dates=True)
test_df = pd.read_csv('data/test.csv', index_col='datetime',parse_dates=True)

print(train_df.shape, val_df.shape, test_df.shape)


# In[52]:


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


# In[53]:


def train_model(model, train_ds, val_ds, patience=5, max_epochs=50):
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    
    model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteError()])
    
    history = model.fit(train_ds, epochs=max_epochs, validation_data=val_ds, callbacks=[early_stopping])
    
    return history


# In[54]:


def plot_history(history):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend(loc='best')

    ax2.plot(history.history['mean_absolute_error'], label='Train')
    ax2.plot(history.history['val_mean_absolute_error'], label='Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
    ax2.set_title('Mean absolute error')
    ax2.legend(loc='best')

    plt.tight_layout()


# In[55]:


def plot_predictions(model, sample_batch, model_type):
    
    inputs, outputs = sample_batch
    preds = model(inputs)
    
    
    if model_type == 'multi_output':
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        
        RD_actual_scaled = outputs.numpy().flatten()[0::2]
        RD_actual = RD_actual_scaled * (RD_max_test - RD_min_test) + RD_min_test
        
        RD_predictions_scaled = preds.numpy().flatten()[0::2]
        RD_predictions = RD_predictions_scaled * (RD_max_train - RD_min_train) + RD_min_train

        R_actual_scaled = outputs.numpy().flatten()[1::2]
        R_actual = R_actual_scaled * (R_max_test - R_min_test) + R_min_test
        
        R_predictions_scaled = preds.numpy().flatten()[1::2]
        R_predictions = R_predictions_scaled * (R_max_train - R_min_train) + R_min_train
        
        ax1.plot(RD_actual, label='Actual')
        ax1.plot(RD_predictions, label='Predicted')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('River Discharge')
        ax1.legend(loc='best')
        ax1.set_title('Predictions on a sample batch')
        
        ax2.plot(RD_actual, label='Actual')
        ax2.plot(RD_predictions, label='Predicted')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Rain Sum')
        ax2.legend(loc='best')
        
    plt.tight_layout()


# Multi-output model

# In[58]:


train_ds_mo = format_timeseries_data(train_df, 3, 3, ['river_discharge', 'rain_sum'])
val_ds_mo = format_timeseries_data(val_df, 3, 3, ['river_discharge', 'rain_sum'])
test_ds_mo = format_timeseries_data(test_df, 3, 3, ['river_discharge', 'rain_sum'])

mo_sample_batch = next(iter(test_ds_mo))


# In[59]:


mo_dnn = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2)
])

mo_dnn_history = train_model(mo_dnn, train_ds_mo, val_ds_mo)



# In[62]:


def plot_evaluation(model_list, mae):
    
    fig, ax = plt.subplots()
    
    ax.bar(model_list, mae, width=0.3)
    ax.set_ylabel('MAE')
    ax.set_xlabel('Models')
    ax.set_ylim(0, max(mae)+0.15)
    for index, value in enumerate(mae):
        ax.text(x=index, y=value+0.005, s=str(round(value, 3)), ha='center')
    
    plt.tight_layout()


# In[64]:


mo_lstm = Sequential([
    LSTM(32, return_sequences=True),
    Dense(2)
])

mo_lstm_history = train_model(mo_lstm, train_ds_mo, val_ds_mo)



# In[ ]:


inputs, outputs = mo_sample_batch
preds = mo_lstm(inputs)

pre_array = preds[0]

np_array = np.array(pre_array)

df_scaled = pd.DataFrame(np_array)

df = df_scaled.rename(columns={0: "river_discharge", 1: "rain_sum"})

df['river_discharge'] = df['river_discharge'].apply(lambda x: x*(RD_max_train - RD_min_train) + RD_min_train)
df['rain_sum'] = df['rain_sum'].apply(lambda x: x*(R_max_train - R_min_train) + R_min_train)


#RD_predictions_scaled = preds.numpy().flatten()[0::2]
#RD_predictions = RD_predictions_scaled * (RD_max_train - RD_min_train) + RD_min_train
        
#R_predictions_scaled = preds.numpy().flatten()[1::2]
#R_predictions = R_predictions_scaled * (R_max_train - R_min_train) + R_min_train
      

#predicted_results = mo_lstm.predict(test_ds_mo)

#predicted_array= predicted_results[0]

#my_array = np.array(predicted_array)

#df = pd.DataFrame(my_array)

