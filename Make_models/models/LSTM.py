import numpy as np
from keras.layers import (LSTM, Activation, Add, BatchNormalization, Conv2D,
                          Dense, Dropout, Input, MaxPooling2D, Reshape)
from keras.models import Model
from keras.optimizers import SGD
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras.backend import tensorflow_backend


class simple_LSTM():
    def __init__(self, config):
        self.config = config
    def conf_model(self):
        input_img = Input(shape=(None,
                                self.config.data_loader.data_size))
        for i in range(self.config.model.layer_num):
            if i == 0:
                x = LSTM(self.config.model.unit_num[i],
                        activation=self.config.model.act,
                        return_sequences=self.config.model.return_sequence[i]
                        )(input_img)
                # x = BatchNormalization()(x)
                x = Dropout(0.15)(x)
            elif i == self.config.model.layer_num - 1:
                x = LSTM(self.config.model.unit_num[i],
                        activation=self.config.model.act,
                        return_sequences=self.config.model.return_sequence[i]
                        )(x)
                x = Dense(self.config.data_loader.data_size)(x)
            else:
                x = LSTM(self.config.model.unit_num[i],
                        activation=self.config.model.act,
                        return_sequences=self.config.model.return_sequence[i]
                        )(x)
                # x = BatchNormalization()(x)
                x = Dropout(0.15)(x)
        x = Activation("linear")(x)
        model = Model(input_img, x)
        print('\n\nModel was created.')
        print('\n----------------Model Configuration----------------\n')
        print('Model                   : LSTM\n')
        print('Input shape of model    : ', input_img.shape)
        print('Output shape of model   : ', x.shape)
        print('Number of layers        : ' + str(self.config.model.layer_num))
        print('Number of units         : ', 
            ", ".join( repr(e) for e in self.config.model.unit_num))
        print('\nOptimizer               : ' + self.config.model.optimizer)
        print('Loss function           : ' + self.config.model.loss)
        print('Activation function     : ' + self.config.model.act)
        print('\n---------------------------------------------------\n')
        
        return model

    def compile_model(self, model):
        if self.config.model.optimizer == 'SGD':
            sgd = SGD(
                lr=self.config.model.lr,
                momentum=self.config.model.momentum,
                decay=self.config.model.decay,
                nesterov=self.config.model.nesterov
            )
            model.compile(
            optimizer=sgd,
            loss=self.config.model.loss
            )
        else:
            model.compile(
            optimizer=self.config.model.optimizer,
            loss=self.config.model.loss
            )
        return model

    def make_model(self):
        if self.config.GPU.multi_gpu:
            config_gpu = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    visible_device_list=self.config.GPU.number
                )
            )
            session = tf.Session(config=config_gpu)
            tensorflow_backend.set_session(session)
            with tf.device("/cpu:0"):
                base_model = self.conf_model()
            model = multi_gpu_model(base_model, gpus=self.config.GPU.gpu_count)
            model = self.compile_model(model)
        else:
            config_gpu = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    visible_device_list=self.config.GPU.number
                )
            )
            session = tf.Session(config=config_gpu)
            tensorflow_backend.set_session(session)
            model = self.conf_model()
            model = self.compile_model(model)
            base_model = None
        return model, base_model

class LSTM_with_shape():
    def __init__(self, config):
        self.config = config

    def conf_model(self):
        act = self.config.model.act
        input_img_CNN = Input(shape=(120, 120, 1))
        y = Conv2D(8, (21, 21), padding='valid')(input_img_CNN) # 100,100,8
        y = BatchNormalization()(y)
        y = Activation(act)(y)
        y = Conv2D(8, (4, 4), padding='valid', strides=(2, 2))(y) # 49,49,8
        y = BatchNormalization()(y)
        y = Activation(act)(y)
        y = Conv2D(8, (3, 3), padding='valid', strides=(2, 2))(y) # 24,24,8
        y = BatchNormalization()(y)
        y = Activation(act)(y)
        y = Conv2D(1, (2, 2), padding='valid', strides=(2, 2))(y) # 12,12,8
        y = BatchNormalization()(y)
        y = Activation(act)(y)
        y = Reshape([144])(y)
        y = Dense(128)(y)
        y = BatchNormalization()(y)
        y = Activation(act)(y)

        input_img = Input(shape=(None,
                                self.config.data_loader.data_size))
        for i in range(self.config.model.layer_num):
            if i == 0:
                x = LSTM(self.config.model.unit_num[i],
                        activation=act,
                        return_sequences=self.config.model.return_sequence[i]
                        )(input_img)
                # x = BatchNormalization()(x)
                x = Dropout(0.15)(x)
            elif i == self.config.model.layer_num - 1:
                x = LSTM(self.config.model.unit_num[i],
                        activation=act,
                        return_sequences=self.config.model.return_sequence[i]
                        )(x)
                x = Dense(self.config.data_loader.data_size)(x)
            else:
                x = LSTM(self.config.model.unit_num[i],
                        activation=act,
                        return_sequences=self.config.model.return_sequence[i]
                        )(x)
                # x = BatchNormalization()(x)
                x = Dropout(0.15)(x)
            if i == self.config.model.shape_input_layer - 1:
                x = Add()([x, y])
        x = Activation("linear")(x)
        model = Model([input_img_CNN, input_img], x)

        print('\n\nModel was created.')
        print('\n----------------Model Configuration----------------\n')
        print('Model                   : LSTM with shape input\n')
        print('Input shape of model    : ', input_img_CNN.shape ,input_img.shape)
        print('Output shape of model   : ', x.shape)
        print('Number of layers        : ' + str(self.config.model.layer_num))
        print('Number of units         : ', 
            ", ".join( repr(e) for e in self.config.model.unit_num))
        print('Layer to be input shape : After ' +
            str(self.config.model.shape_input_layer) + 'st layer')
        print('\nOptimizer               : ' + self.config.model.optimizer)
        print('Loss function           : ' + self.config.model.loss)
        print('Activation function     : ' + self.config.model.act)
        print('\n---------------------------------------------------\n')

        return model

    def compile_model(self, model):
        if self.config.model.optimizer == 'SGD':
            sgd = SGD(
                lr=self.config.model.lr,
                momentum=self.config.model.momentum,
                decay=self.config.model.decay,
                nesterov=self.config.model.nesterov
            )
            model.compile(
            optimizer=sgd,
            loss=self.config.model.loss
            )
        else:
            model.compile(
            optimizer=self.config.model.optimizer,
            loss=self.config.model.loss
            )

        return model
    
    def make_model(self):
        if self.config.GPU.multi_gpu:
            config_gpu = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    visible_device_list=self.config.GPU.number
                )
            )
            session = tf.Session(config=config_gpu)
            tensorflow_backend.set_session(session)
            with tf.device("/cpu:0"):
                base_model = self.conf_model()
            model = multi_gpu_model(base_model, gpus=self.config.GPU.gpu_count)
            model = self.compile_model(model)
        else:
            config_gpu = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    visible_device_list=self.config.GPU.number
                )
            )
            session = tf.Session(config=config_gpu)
            tensorflow_backend.set_session(session)
            model = self.conf_model()
            model = self.compile_model(model)
            base_model = None
        return model, base_model
