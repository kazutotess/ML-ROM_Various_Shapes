import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                          Add, BatchNormalization, Activation)
from keras.models import Model
from keras.backend import tensorflow_backend
from keras.callbacks import ModelCheckpoint
from keras import backend as K


def conv_down_block(input_img, size, layer_nm, chanel_nm, act):
    for i in range(layer_nm[0]):
        if i == 0:
            x = Conv2D(chanel_nm[0],
                       (size, size),
                       padding='same')(input_img)
        else:
            x = Conv2D(chanel_nm[0],
                       (size, size),
                       padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    for i in range(layer_nm[1]):
        x = Conv2D(chanel_nm[1],
                   (size, size),
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    for i in range(layer_nm[2]):
        x = Conv2D(chanel_nm[2],
                   (size, size),
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        if i != layer_nm[2] - 1:
            x = MaxPooling2D((2, 2), padding='same')(x)
    return x


def conv_up_block(encoded, size, layer_nm, chanel_nm, act, phys_num):
    for i in range(layer_nm[-1] - 1):
        if i == 0:
            x = Conv2D(chanel_nm[-1],
                       (size, size),
                       padding='same')(encoded)
        else:
            x = Conv2D(chanel_nm[-1],
                       (size, size),
                       padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = UpSampling2D((2, 2))(x)  # 12,6,4

    for i in range(layer_nm[-2]):
        x = Conv2D(chanel_nm[-2],
                   (size, size),
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = UpSampling2D((2, 2))(x)  # 24,12,8

    for i in range(layer_nm[-3]):
        x = Conv2D(
            chanel_nm[-3], (size, size), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = UpSampling2D((2, 2))(x)  # 384,192,16
    x = Conv2D(phys_num, (size, size),
               activation='linear', padding='same')(x)
    return x


def MS_CNN_AE(x_num, y_num, phys_num, filsize, layer_nm, chanel_nm, act,
              optimizer, loss):
    input_img = Input(
        shape=(
            x_num,
            y_num,
            phys_num)
    )

    filsize1 = filsize[0]
    filsize2 = filsize[1]
    filsize3 = filsize[2]

    conv1 = conv_down_block(
        input_img, filsize1, layer_nm, chanel_nm, act)
    conv2 = conv_down_block(
        input_img, filsize2, layer_nm, chanel_nm, act)
    conv3 = conv_down_block(
        input_img, filsize3, layer_nm, chanel_nm, act)
    x = Add()([conv1, conv2, conv3])
    x = Conv2D(chanel_nm[2], (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    encoded = Activation(act)(x)
    x = Conv2D(chanel_nm[2], (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    conv4 = conv_up_block(x, filsize1, layer_nm, chanel_nm, act, phys_num)
    conv5 = conv_up_block(x, filsize2, layer_nm, chanel_nm, act, phys_num)
    conv6 = conv_up_block(x, filsize3, layer_nm, chanel_nm, act, phys_num)
    decoded = Add()([conv4, conv5, conv6])

    print('\n\nModel was created.')
    print('\n----------------Model Configuration----------------\n')
    print('Model                   : Multi-scale CNN\n')
    print('Input shape of model    : %d, %d, %d'
          % (input_img.shape[-3],
             input_img.shape[-2],
             input_img.shape[-1]))
    print('Shape of encoded data   :   %d,   %d, %d'
          % (encoded.shape[-3],
             encoded.shape[-2],
             encoded.shape[-1]))
    print('Output shape of model   : %d, %d, %d'
          % (decoded.shape[-3],
             decoded.shape[-2],
             decoded.shape[-1]))
    print('\nOptimizer               : ' + optimizer)
    print('Loss function           : ' + loss)
    print('Activation function     : ' + act)
    print('\n---------------------------------------------------\n')

    model = Model(input_img, decoded)

    model.compile(
        optimizer=optimizer,
        loss=loss
    )

    return model


def main():
    # specify GPU
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            visible_device_list="2"
        )
    )
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    # set parameters
    kind_num = 2  # number of shapes
    num_of_ts = 500  # number of instantaneous fields for each shapes
    x_num = 384  # grid point of x direction
    y_num = 192  # grid point of y direction
    phys_num = 3  # u, v, p

    path_to_present_dir = './'  # directory which contains flow data
    save_file = '/CNN_autoencoder/'  # directory for saving ML model
    model_name = 'Test_CNN_AE'  # name of ML model file
    # the model will be saved as
    # path_to_present_dir + save_file + 'Model/' + model_name + '.hdf5'

    act = 'relu'  # activation function
    filsize = [3, 5, 9]  # filter size for each scale CNN
    layer_nm = [1, 4, 2]  # number of layers
    chanel_nm = [16, 8, 4]  # number of chanels
    loss = 'gdl_mse'  # loss function
    optimizer = 'adam'  # optimizer
    ratio_tr_te = 0.2  # ratio of training and validation data
    num_epochs = 2  # number of epochs
    batch_size = 50  # batch size

    # prepare flow data
    X = np.zeros([kind_num, num_of_ts,
                  x_num, y_num, phys_num])

    for i in tqdm(range(1, kind_num + 1)):
        fnstr = path_to_present_dir + '/data/pickles/data_' + \
            '{0:03d}'.format(i)+'.pickle'

        # Pickle load

        with open(fnstr, 'rb') as f:
            obj = pickle.load(f)
        X[i - 1, :, :, :, :phys_num] = obj[:num_of_ts]

    X = np.reshape(X, [-1, X.shape[-3], X.shape[-2], X.shape[-1]])

    x_train, x_test, y_train, y_test = \
        train_test_split(X, X[:, :, :, :phys_num], test_size=ratio_tr_te,
                         random_state=None)

    # construct machine learning model (Multi-Scale CNN AE)
    model = MS_CNN_AE(x_num, y_num, phys_num, filsize,
                      layer_nm, chanel_nm, act, optimizer, loss)

    # train the model
    callbacks = []

    os.makedirs(path_to_present_dir + save_file + 'Model/', exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            path_to_present_dir + save_file + 'Model/' + model_name + '.hdf5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    )

    print('\n-----------------Training Condition----------------\n')
    print('X training data         : ', x_train.shape)
    print('Y training data         : ', y_train.shape)
    print('X test data             : ', x_test.shape)
    print('Y test data             : ', y_test.shape)
    print('Callbacks               : Model Checkpoint')
    print('\n---------------------------------------------------\n')

    print('Training is now begining.')

    history = model.fit(
        x_train,
        y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    df_results = pd.DataFrame(history.history)
    df_results['epoch'] = history.epoch
    os.makedirs(
        path_to_present_dir + save_file + 'History/',
        exist_ok=True
    )
    df_results.to_csv(
        path_or_buf=path_to_present_dir +
        save_file +
        'History/' +
        model_name +
        '.csv',
        index=False
    )
    print('History was saved.')

    K.clear_session()
    print('The session was cleared.')


if __name__ == '__main__':
    main()
