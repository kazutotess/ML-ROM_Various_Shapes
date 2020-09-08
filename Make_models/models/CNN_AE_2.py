from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                          Add, BatchNormalization, Activation, Reshape,
                          Dense, Conv2DTranspose)
from keras.models import Model
from keras.optimizers import SGD
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras.backend import tensorflow_backend

def conv_down_block(input_img, size, config):
    for i in range(config.model.layer_nm[0]):
        if i == 0:
            x = Conv2D(config.model.chanel_nm[0],
                        (size, size),
                        padding='same')(input_img)
        else:
            x = Conv2D(config.model.chanel_nm[0],
                        (size, size),
                        padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(config.model.act)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    for i in range(config.model.layer_nm[1]):
        x = Conv2D(config.model.chanel_nm[1],
                    (size, size),
                    padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(config.model.act)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    for i in range(config.model.layer_nm[2]):
        x = Conv2D(config.model.chanel_nm[2],
                    (size, size),
                    padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(config.model.act)(x)
        if i != config.model.layer_nm[2] - 1:
            x = MaxPooling2D((2, 2), padding='same')(x)
    return x

def conv_up_block(encoded, size, config):
    for i in range(config.model.layer_nm[-1] - 1):
        if i == 0:
            x = Conv2D(config.model.chanel_nm[-1],
                        (size, size),
                        padding='same')(encoded)
        else:
            x = Conv2D(config.model.chanel_nm[-1],
                        (size, size),
                        padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(config.model.act)(x)
        x = UpSampling2D((2, 2))(x)  # 12,6,4

    for i in range(config.model.layer_nm[-2]):
        x = Conv2D(config.model.chanel_nm[-2],
                    (size, size),
                    padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(config.model.act)(x)
        x = UpSampling2D((2, 2))(x)  # 24,12,8

    for i in range(config.model.layer_nm[-3]):
        x = Conv2D(
            config.model.chanel_nm[-3], (size, size), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(config.model.act)(x)
        x = UpSampling2D((2, 2))(x)  # 384,192,16
    x = Conv2D(config.data_loader.phys_num, (size, size),
                activation='linear', padding='same')(x)
    return x

class MS_CNN_AE():
    def __init__(self, config):
        self.config = config

    def conf_model(self):
        input_img = Input(
                shape=(
                    self.config.data_loader.x_num,
                    self.config.data_loader.y_num,
                    self.config.data_loader.phys_num)
            )
        if self.config.data_loader.with_shape:
            input_img = Input(
                    shape=(
                        self.config.data_loader.x_num,
                        self.config.data_loader.y_num,
                        self.config.data_loader.phys_num + 1)
                )
        
        filsize1 = self.config.model.filsize[0]
        filsize2 = self.config.model.filsize[1]
        filsize3 = self.config.model.filsize[2]

        conv1 = conv_down_block(
            input_img, filsize1, self.config)
        conv2 = conv_down_block(
            input_img, filsize2, self.config)
        conv3 = conv_down_block(
            input_img, filsize3, self.config)
        x = Add()([conv1, conv2, conv3])
        x = Conv2D(
            self.config.model.convert_part_chanel_nm, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        ##############################
        if self.config.model.less_than_72:
            x = Activation(self.config.model.act)(x)
            x = Conv2D(self.config.model.convert_part_chanel_nm, (3, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation(self.config.model.act)(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
        ##############################
        encoded = Activation(self.config.model.act)(x)
        ##############################
        if self.config.model.less_than_72:
            x = UpSampling2D((2, 2))(encoded)
            x = Activation(self.config.model.act)(x)
            x = Conv2DTranspose(self.config.model.convert_part_chanel_nm,
                               (3, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation(self.config.model.act)(x)
        ##############################
            x = Conv2D(
                self.config.model.convert_part_chanel_nm, (3, 3), padding='same')(x)
        else:
            x = Conv2D(
                self.config.model.convert_part_chanel_nm, (3, 3), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Activation(self.config.model.act)(x)
        conv4 = conv_up_block(x, filsize1, self.config)
        conv5 = conv_up_block(x, filsize2, self.config)
        conv6 = conv_up_block(x, filsize3, self.config)
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
        print('\nOptimizer               : ' + self.config.model.optimizer)
        print('Loss function           : ' + self.config.model.loss)
        print('Activation function     : ' + self.config.model.act)
        print('\n---------------------------------------------------\n')
        
        model = Model(input_img, decoded)
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