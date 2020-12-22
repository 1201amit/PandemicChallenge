import pandas as pd
import csv
import os 
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from ML_mobility import mobility_model
import keras


def train_mobility(path_to_data,lookback_days=60):
    train_df = pd.read_csv(path_to_data)
    train_df=train_df.to_numpy()
    total_x=train_df[:, 2:-5]
    total_y=train_df[:,-5:]
    model_dir="models"
#     print(total_y.shape)
    
    total_x = tf.convert_to_tensor(np.asarray(total_x, dtype=np.float64))
    total_y = tf.convert_to_tensor(np.asarray(total_y, dtype=np.float64))
    callbacks_list = [tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1)]

    model=mobility_model(lookback_days)
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='mae', optimizer=opt, metrics=["mean_squared_error"])
    
    
    model.fit(total_x,total_y, batch_size=32,validation_split=0.2, epochs = 800,callbacks=callbacks_list, verbose=1)
    print(model.summary())

def predict_mobility(npi_test,lookback_days=60):
    model_dir="models"
    npi_test=npi_test[np.newaxis,:]
    npi_test = tf.convert_to_tensor(np.asarray(npi_test, dtype=np.float64))


    
    model=mobility_model(lookback_days)
    
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    
    model.load_weights(latest_checkpoint)
    
    mobility= model.predict(npi_test)
    return mobility

        
    