import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import (LSTM, Activation, Add, BatchNormalization, Conv2D,
                          Dense, Dropout, Input, Reshape)
from keras.models import Model
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K


def LSTM_with_shape(act, data_size, layer_num, unit_num, return_sequence,
                    shape_input_layer, optimizer, loss):
    input_img_CNN = Input(shape=(120, 120, 1))
    y = Conv2D(8, (21, 21), padding='valid')(input_img_CNN)  # 100,100,8
    y = BatchNormalization()(y)
    y = Activation(act)(y)
    y = Conv2D(8, (4, 4), padding='valid', strides=(2, 2))(y)  # 49,49,8
    y = BatchNormalization()(y)
    y = Activation(act)(y)
    y = Conv2D(8, (3, 3), padding='valid', strides=(2, 2))(y)  # 24,24,8
    y = BatchNormalization()(y)
    y = Activation(act)(y)
    y = Conv2D(1, (2, 2), padding='valid', strides=(2, 2))(y)  # 12,12,8
    y = BatchNormalization()(y)
    y = Activation(act)(y)
    y = Reshape([144])(y)
    y = Dense(128)(y)
    y = BatchNormalization()(y)
    y = Activation(act)(y)

    input_img = Input(shape=(None, data_size))
    for i in range(layer_num):
        if i == 0:
            x = LSTM(unit_num[i],
                     activation=act,
                     return_sequences=return_sequence[i]
                     )(input_img)
            x = Dropout(0.15)(x)
        elif i == layer_num - 1:
            x = LSTM(unit_num[i],
                     activation=act,
                     return_sequences=return_sequence[i]
                     )(x)
            x = Dense(data_size)(x)
        else:
            x = LSTM(unit_num[i],
                     activation=act,
                     return_sequences=return_sequence[i]
                     )(x)
            x = Dropout(0.15)(x)
        if i == shape_input_layer - 1:
            x = Add()([x, y])
    x = Activation("linear")(x)
    model = Model([input_img_CNN, input_img], x)

    print('\n\nModel was created.')
    print('\n----------------Model Configuration----------------\n')
    print('Model                   : LSTM with shape input\n')
    print('Input shape of model    : ',
          input_img_CNN.shape, input_img.shape)
    print('Output shape of model   : ', x.shape)
    print('Number of layers        : ' + str(layer_num))
    print('Number of units         : ',
          ", ".join(repr(e) for e in unit_num))
    print('Layer to be input shape : After ' +
          str(shape_input_layer) + 'st layer')
    print('\nOptimizer               : ' + optimizer)
    print('Loss function           : ' + loss)
    print('Activation function     : ' + act)
    print('\n---------------------------------------------------\n')

    model.compile(optimizer=optimizer, loss=loss)

    return model


def main():
    # specify GPU
    # you need to coment out this part if you don't use GPU
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            visible_device_list="3"
        )
    )
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    # set parameters
    num_of_ts = 500  # number of training data for each shape
    num_of_ts_for_data = 550  # number of instance to generate training data
    number_of_shape = 80  # number of shapes
    maxlen = 20  # number of input time steps
    time_step = 1  # time step size
    data_size = 72  # size of latent vector

    path_to_present_dir = './'  # directory which contains flow data
    dataset_name = '72_values_MS-BN-1_dataset.csv'  # data file name
    # path for data file
    path_data = path_to_present_dir + 'data/LSTM/Dataset/' + dataset_name
    save_file = 'LSTM/'  # directory for saving ML model
    model_name = 'Test_LSTM'  # name of ML model files

    layer_num = 3  # number of LSTM layer
    unit_num = [128, 128, 128]  # number of units for each layer
    # whether the LSTM layers return sequential output or not
    return_sequence = [True, True, True]
    shape_input_layer = 1  # layer number of shape input

    ratio_tr_te = 0.2  # ratio of training and validation data
    act = 'tanh'  # activation function
    optimizer = 'adam'  # optimizer
    loss = 'mse'  # loss function
    num_epochs = 2  # number of epochs
    batch_size = 1000  # batch size

    # perpare data
    assert num_of_ts + time_step * (maxlen - 1) < \
        num_of_ts_for_data, 'The data aumont is not enough.'

    data_LSTM = pd.read_csv(path_data, header=None, delim_whitespace=False)
    data_LSTM = data_LSTM.values

    X_CNN = np.zeros([number_of_shape * num_of_ts, 120, 120, 1])
    for i in range(number_of_shape):
        data_CNN = pd.read_csv(
            path_to_present_dir +
            '/data/LSTM/Flags/Flag' +
            '{0:03d}'.format(i + 1) + '.csv',
            header=None,
            delim_whitespace=False
        )
        data_CNN = data_CNN.values
        X_CNN[i * num_of_ts: (i + 1) * num_of_ts, :, :, 0] = data_CNN

    X = np.zeros([number_of_shape * num_of_ts, maxlen, data_size])
    Y = np.zeros([number_of_shape * num_of_ts, maxlen, data_size])

    for i in range(number_of_shape):
        for j in range(num_of_ts):
            X[i * num_of_ts + j] = \
                data_LSTM[
                    i * num_of_ts_for_data + j:
                i * num_of_ts_for_data + j +
                time_step * maxlen: time_step
            ]
            Y[i * num_of_ts + j] = \
                data_LSTM[
                    i * num_of_ts_for_data + j + 1:
                i * num_of_ts_for_data + j +
                time_step * maxlen + 1: time_step
            ]

    X_CNN_train, X_CNN_test, X_train, X_test, y_train, y_test = \
        train_test_split(X_CNN,
                         X,
                         Y,
                         test_size=ratio_tr_te,
                         random_state=None)
    x_train = [X_CNN_train, X_train]
    x_test = [X_CNN_test, X_test]

    # construct machine learning model (LSTM with shape)
    model = LSTM_with_shape(act, data_size, layer_num, unit_num,
                            return_sequence, shape_input_layer,
                            optimizer, loss)

    # train the model
    callbacks = []

    # model save
    os.makedirs(path_to_present_dir + save_file + 'Model/', exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            path_to_present_dir + save_file + 'Model/' + model_name + '.hdf5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    )

    # history save
    os.makedirs(
        path_to_present_dir + save_file + 'History/',
        exist_ok=True
    )
    callbacks.append(
        CSVLogger(path_to_present_dir +
                  save_file +
                  'History/' +
                  model_name +
                  '.csv',)
    )

    print('\n-----------------Training Condition----------------\n')
    print('X training data         : ', X_train.shape)
    print('Y training data         : ', y_train.shape)
    print('X test data             : ', X_test.shape)
    print('Y test data             : ', y_test.shape)
    print('Callbacks               : Model Checkpoint')
    print('\n---------------------------------------------------\n')

    print('Training is now begining.')

    model.fit(
        x_train,
        y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    K.clear_session()
    print('The session was cleared.')


if __name__ == '__main__':
    main()
