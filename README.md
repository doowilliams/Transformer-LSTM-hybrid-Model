# Transformer-LSTM Hybrid Model for Time Series Forecasting

## Overview

This project involves developing a hybrid Transformer-LSTM model to predict hourly solar power yield based on a variety of environmental and temporal features. The dataset consists of historical data with multiple environmental variables and the target variable of total power yield. The workflow includes data preprocessing, feature engineering, model building, training, evaluation, and visualization.

## Project Structure

1. **Data Loading and Preprocessing**
2. **Feature Engineering**
3. **Data Scaling and Splitting**
4. **Model Definition**
5. **Model Training**
6. **Evaluation**
7. **Visualization**
8. **Model Export**

## 1. Data Loading and Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('Durning centre data hourly captured - with enviromental features1.csv')

# Parse datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)
df.drop(columns=['Date', 'Time', 'Precipitation', 'Wind Speed (10 meters)'], inplace=True)
```

## 2. Feature Engineering

```python
# Add temporal features
df['Hour'] = df.index.hour
df['Day'] = df.index.day
df['Weekday'] = df.index.weekday
df['Month'] = df.index.month
df['Year'] = df.index.year

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)
season_dummies = pd.get_dummies(df['Season'], prefix='Season')
df = pd.concat([df, season_dummies], axis=1)

# Rolling statistics
df['pv_roll_mean_3'] = df['Total Yield[kWh]'].rolling(window=3).mean()
df['pv_roll_std_3'] = df['Total Yield[kWh]'].rolling(window=3).std()

# Lag features
df['temp_lag_1'] = df['Temprature (2 meters)'].shift(1)
df['wind_lag_1'] = df['Wind Direction (10 meters)'].shift(1)
df['solar_lag_1'] = df['Solar Irradiance'].shift(1)

# Fourier Transform features
df['sin_hour'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['Hour'] / 24)

df.fillna(0, inplace=True)
df.drop(columns=['Season'], inplace=True)
```

## 3. Data Scaling and Splitting

```python
# Prepare data for scaling
X = df.drop(columns=['Total Yield[kWh]']).values.astype('float32')
y = df['Total Yield[kWh]'].values.astype('float32').reshape(-1, 1)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape X to 3D array for LSTM input
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
```

## 4. Model Definition

### Transformer Encoder Block

```python
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Conv1D, Dropout, Input
from tensorflow.keras.models import Model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    x = Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res
```

### Hybrid Transformer-LSTM Model

```python
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization

def build_transformer_LSTM_model(input_shape, num_heads, head_size, encoder_decoder_dropout, num_cnn_layers, ffd_units, ffd_dropout, lstm_units, num_lstm_layers):
    inputs = Input(shape=input_shape)
    x = inputs

    # Stacked LSTM layers
    for _ in range(num_lstm_layers - 1):
        x = LSTM(lstm_units, return_sequences=True)(x)
        x = Dropout(encoder_decoder_dropout)(x)
    x = LSTM(lstm_units, return_sequences=True)(x)

    # Transformer Encoder layers
    for _ in range(num_cnn_layers):
        x = transformer_encoder(x, head_size, num_heads, ffd_units[0], encoder_decoder_dropout)

    # Fully connected layers
    for units in ffd_units:
        x = Dense(units, activation='relu')(x)
        x = Dropout(ffd_dropout)(x)
        x = BatchNormalization()(x)

    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model
```

## 5. Model Training

```python
import tensorflow as tf

def compile_and_fit(model, xtrain=X_train, ytrain=y_train, xtest=X_test, ytest=y_test, epochs=50, learning_rate=0.001):
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
            tf.keras.metrics.MeanAbsoluteError(name='mae')
        ]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    history = model.fit(
        xtrain, ytrain,
        validation_data=(xtest, ytest),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler]
    )

    return history

# Train the model
num_heads = 4
head_size = 64
encoder_decoder_dropout = 0.1
num_cnn_layers = 4
ffd_units = [282]
ffd_dropout = 0.2
lstm_units = 32
num_lstm_layers = 4
input_shape = (X_train.shape[1], X_train.shape[2])

model = build_transformer_LSTM_model(
    input_shape, num_heads, head_size, encoder_decoder_dropout,
    num_cnn_layers, ffd_units, ffd_dropout, lstm_units, num_lstm_layers
)

history = compile_and_fit(model, epochs=30, learning_rate=0.01)
```

## 6. Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predict
y_pred = model.predict(X_test).flatten()
y_test_flat = y_test.flatten()

# Metrics
mse = mean_squared_error(y_test_flat, y_pred)
mae = mean_absolute_error(y_test_flat, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_flat, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2 score):", r2)
```

## 7. Visualization

### Matplotlib

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
```

### Plotly

```python
import plotly.express as px

# Test data index
test_index = df.index[-len(y_test):]

# Reshape and inverse transform for plotting
y_test_2d = y_test

.reshape(-1, 1)
y_pred_2d = y_pred.reshape(-1, 1)

y_test_inv = scaler_y.inverse_transform(y_test_2d)
y_pred_inv = scaler_y.inverse_transform(y_pred_2d)

# Plotly plot
fig = px.line(x=test_index, y=[y_test_inv.flatten(), y_pred_inv.flatten()],
              labels={'x': 'Date', 'value': 'Total Yield'},
              title='Actual vs Predicted Total Yield',
              markers=True)

fig.add_scatter(x=test_index, y=y_test_inv.flatten(), mode='lines+markers', name='Actual')
fig.add_scatter(x=test_index, y=y_pred_inv.flatten(), mode='lines+markers', name='Predicted')

fig.update_layout(xaxis_title='Datetime', yaxis_title='Total Yield')
fig.show()
```

## 8. Model Export

```python
model.save('transformer_lstm_model.h5')
print("Model saved as 'transformer_lstm_model.h5'")
```

## Summary

This code combines Transformer and LSTM components to build a robust model for time series forecasting, specifically aimed at predicting solar power yield. The model incorporates advanced techniques such as temporal feature engineering, data scaling, and model callbacks to optimize performance.
