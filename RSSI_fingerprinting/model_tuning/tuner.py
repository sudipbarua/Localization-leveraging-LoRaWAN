import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import keras_tuner as kt

x_w_train = pd.read_csv('RSSI_fingerprinting/files/x_w_train.csv', index_col=0)
x_w_test = pd.read_csv('RSSI_fingerprinting/files/x_w_test.csv', index_col=0)
y_w_train = pd.read_csv('RSSI_fingerprinting/files/y_w_train.csv', index_col=0)
y_w_test = pd.read_csv('RSSI_fingerprinting/files/y_w_test.csv', index_col=0)

x_w_train = x_w_train.head(100)
y_w_train = y_w_train.head(100)

# reshape input data
x_train = x_w_train.drop(columns=['timestamp', 'gw_ref'])
x_train_reshaped = x_train.values.reshape(x_train.shape[0], x_train.shape[1])
y_train = y_w_train.values.reshape(y_w_train.shape[0], y_w_train.shape[1])

x_test = x_w_test.drop(columns=['timestamp', 'gw_ref'])
x_test_reshaped = x_test.values.reshape(x_test.shape[0], x_test.shape[1])
y_test = y_w_test.values.reshape(y_w_test.shape[0], y_w_test.shape[1])


def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(x_train_reshaped.shape[1],)))

  # Choose the number of layers
  hp_layers = hp.Int('layers', min_value=2, max_value=3, step=1)
  
  for layer in range(hp_layers):
    hp_activation = hp.Choice(f'activation{layer}', values=['relu', 'leaky_relu', 'elu'])
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int(f'units{layer}', min_value=32, max_value=512, step=8)
    model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))
  
  # Output layer
  model.add(keras.layers.Dense(2, activation='linear'))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mean_squared_error', metrics=['mse'])

  return model


tuner = kt.Hyperband(model_builder,
                     objective='val_mse',
                     max_epochs=10,
                     factor=3,
                     directory='RSSI_fingerprinting/model_tuning',
                     project_name='hyperband_tuner')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


tuner.search(x_train_reshaped, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of layers is {best_hps.get('layers')}.
""")

"""
*******For Linux users**********
Tuning takes hours (if not days or weeks). So it will be necessary to run this program on background.
So we create a seperate screen for running an instance of this. This way it will neither be interrupted by other actvities
nor terminated when the session is closed.
Simply type "screen -S tuning-1" to create a new session and run the program.
And if a screen is already running , we can use "screen -r <session name>" command to check the running session and switch to it.
Note: NEVER PRESS CTRL+D/C when you are in a running screen. It will terminate the screen.
INSTEAD press CTRL+A+D to detach the screen
"""