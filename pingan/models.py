from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Merge, InputLayer
from keras.layers.normalization import BatchNormalization


def create_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=128, input_shape=input_shape, activation='tanh', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=128, activation='tanh', dropout=0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    return model


def create_lstm_cnn(input_shape):
    # LSTM part
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=128, input_shape=input_shape, activation='tanh', return_sequences=True))
    model_lstm.add(LSTM(units=128, activation='tanh', dropout=0.5))
    print("here lstm:", model_lstm.output_shape)
    model_lstm.add(BatchNormalization())

    # CNN part
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=128, kernel_size=5, padding='valid', input_shape=input_shape, activation='relu'))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPooling1D(pool_size=3))
    model_cnn.add(Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu'))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPooling1D(pool_size=5))
    model_cnn.add(Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPooling1D(pool_size=model_cnn.output_shape[1]))
    model_cnn.add(Flatten())
    model_cnn.add(Dropout(0.5))

    # Merge all part
    model = Sequential()
    model.add(Merge([model_lstm, model_cnn], mode='concat'))
    model.add(Dense(128, activation='relu'))
    model_cnn.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(1))

    return model

""" 0.053
def create_lstm_cnn(input_shape):
    # LSTM part
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=128, input_shape=input_shape, activation='tanh', return_sequences=True))
    print("here lstm:", model_lstm.output_shape)
    model_lstm.add(LSTM(units=128, activation='tanh', dropout=0.5))

    # CNN part
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=128, kernel_size=5, padding='valid', input_shape=input_shape, activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=3))
    model_cnn.add(Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=5))
    model_cnn.add(Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=model_cnn.output_shape[1]))
    model_cnn.add(Flatten())
    model_cnn.add(Dropout(0.5))

    # Merge all part
    model = Sequential()
    model.add(Merge([model_lstm, model_cnn], mode='concat'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(1))

    return model
"""


def create_cnn_dense(trip_input_shape, user_input_shape):
    # CNN part
    model_cnn = Sequential()
    cnn_input = InputLayer(input_shape=trip_input_shape)
    model_cnn.add(cnn_input)
    model_cnn.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=3))
    model_cnn.add(Dropout(0.6))
    model_cnn.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Conv1D(filters=128, kernel_size=2, padding='valid', activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=model_cnn.output_shape[1]))
    print('cnn:', model_cnn.output_shape)
    model_cnn.add(Flatten())
    print('cnn:', model_cnn.output_shape)

    # Dense network part
    model_dense = Sequential()
    dense_input = InputLayer(input_shape=user_input_shape)
    model_dense.add(dense_input)
    model_dense.add(Dense(units=128, activation='relu'))
    model_dense.add(Dense(units=128, activation='relu'))
    model_dense.add(BatchNormalization())
    model_dense.add(Dense(units=64, activation='relu'))
    print('dense:',model_dense.output_shape)
    #model_dense.add(Flatten())

    model = Sequential()
    model.add(Merge([model_cnn, model_dense], mode='concat'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(1))

    return model_cnn

