from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense,Dropout
from keras.optimizers import Adam


def create_model(input_shape, drop_out=0.3, act='tanh'):
    model = Sequential()
    model.add(LSTM(units=256, input_shape=input_shape, activation=act, recurrent_activation=act, return_sequences=True))
    model.add(LSTM(units=128, activation=act, recurrent_activation=act, return_sequences=False))
    model.add(Dense(units=128, activation=act))
    model.add(Dropout(rate=drop_out))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model
