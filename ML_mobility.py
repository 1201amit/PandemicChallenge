import pandas as pd
import csv
import keras.backend as K
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.constraints import Constraint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
# from keras.layers import Input
# from keras.layers import LSTM
# from keras.layers import Lambda
from keras import Model
from keras import layers
from keras import Input

class mobility_model(Model):
    def __init__(self, lookback_days=60):
        super(mobility_model, self).__init__()
        self.dense1= Dense(units=256, name='dense1')
#         self.BN1= BatchNormalization()
        self.dropout1= Dropout(0.5)
        self.dense2= Dense(units=64, name='dense2')
        self.dropout2=Dropout(0.5)
        self.dense3= Dense(units=32, name='dense3')
        self.mobility_output = Dense(units=5,
                                   name='final_output')
    def call(self,input_, training=False):
        x=self.dense1(input_)
#         x=self.BN1(x)
        x=self.dropout1(x)
        x=self.dense2(x)
        x=self.dropout2(x)
        x=self.dense3(x)
        x=self.mobility_output(x) 
        return x