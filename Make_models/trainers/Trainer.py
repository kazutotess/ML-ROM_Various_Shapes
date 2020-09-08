import os
import time
import pandas as pd
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)
from keras import backend as K 
from utils.utils import MultiGPUCheckpointCallback
from sklearn.model_selection import KFold
import numpy as np
import sys

class trainer():
    def __init__(self, model, data, config):
        self.model, self.base_model = model.make_model()
        self.data = data
        self.config = config
        self.callbacks = []
        self.val_loss_time = {
            'val_loss': [],
            'time': []
        }
        self.init_callbacks()

    def init_callbacks(self):
        if self.config.callbacks.modelcheckpoint and \
            not self.config.GPU.multi_gpu:
            os.makedirs(
                self.config.data_loader.path_to_present_dir +
                self.config.callbacks.save_file +
                'Model/',
                exist_ok=True
            )
            self.callbacks.append(
                ModelCheckpoint(
                    self.config.data_loader.path_to_present_dir +
                    self.config.callbacks.save_file +
                    'Model/' +
                    self.config.callbacks.model_name +
                    '.hdf5',
                    monitor=self.config.callbacks.cp_monitor,
                    save_best_only=self.config.callbacks.cp_save_best_only,
                    verbose=self.config.callbacks.cp_verbose
                )
            )
        if self.config.callbacks.modelcheckpoint and \
            self.config.GPU.multi_gpu:
            os.makedirs(
                self.config.data_loader.path_to_present_dir +
                self.config.callbacks.save_file +
                'Model/',
                exist_ok=True
            )
            self.callbacks.append(
                MultiGPUCheckpointCallback(
                    self.config.data_loader.path_to_present_dir +
                    self.config.callbacks.save_file +
                    'Model/' +
                    self.config.callbacks.model_name +
                    '.hdf5',
                    self.base_model,
                    monitor=self.config.callbacks.cp_monitor,
                    save_best_only=self.config.callbacks.cp_save_best_only,
                    verbose=self.config.callbacks.cp_verbose
                )
            )
        if self.config.callbacks.earlystopping:
            self.callbacks.append(
                EarlyStopping(
                    monitor=self.config.callbacks.es_monitor,
                    patience=self.config.callbacks.es_patience,
                    verbose=self.config.callbacks.es_verbose
                )
            )

        if self.config.callbacks.tensorboard:
            self.callbacks.append(
                TensorBoard(
                    log_dir=self.config.callbacks.tensorboard_log_dir,
                    write_graph=self.config.callbacks.tensorboard_write_graph,
                )
            )

        if self.config.model.optimizer == 'SGD' and \
                self.config.callbacks.reduceLR:
            self.callbacks.append(
                ReduceLROnPlateau(
                    monitor=self.config.callbacks.reduceLR_monitor,
                    factor=self.config.callbacks.reduceLR_factor,
                    patience=self.config.callbacks.reduceLR_patience,
                    verbose=self.config.callbacks.reduceLR_verbose,
                )
            )

    def train(self):
        if self.config.data_loader.generator:
            self.x_test, self.y_test = \
                self.data.make_val_data()

            print('\n-----------------Training Condition----------------\n')
            if self.config.GPU.multi_gpu:
                print('This model will be trained using ' + str(self.config.GPU.gpu_count) + ' GPUs.')
            print('This model will be trained by fit_generator.')
            print('Number of epochs        : ' + str(self.config.trainer.num_epochs))
            if type(self.x_test) is not list:
                print('X test data             : ', self.x_test.shape)
            if type(self.y_test) is not list:
                print('Y test data             : ', self.y_test.shape)
            if self.config.callbacks.modelcheckpoint:
                print('Callbacks               : Model Checkpoint')
            if self.config.callbacks.earlystopping:
                print('                        : Early Stopping')
            if self.config.callbacks.tensorboard:
                print('                        : Tensor Board')
            if self.config.model.optimizer == 'SGD' and \
                    self.config.callbacks.reduceLR:
                print('                        : Reduce Learning Rate On Plateau')
            if self.config.callbacks.earlystopping:
                print('Patiance of ES          : ' + str(self.config.callbacks.es_patience))
            if self.config.model.optimizer == 'SGD' and \
                    self.config.callbacks.reduceLR:
                print('Patiance of RLRP        : ' + str(self.config.callbacks.reduceLR_patience))
            print('\n---------------------------------------------------\n')

            print('Training is now begining.')

            start = time.time()
            history = self.model.fit_generator(
                self.data,
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose,
                callbacks=self.callbacks,
                validation_data=(
                    self.x_test,
                    self.y_test
                ),
                max_queue_size=15,
                workers=10,
                use_multiprocessing=True,
                shuffle=self.config.trainer.shuffle,
            )
            elapsed_time = time.time() - start

        else:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                self.data.test_train_split()

            print('\n-----------------Training Condition----------------\n')
            if self.config.GPU.multi_gpu:
                print('This model will be trained using ' + str(self.config.GPU.gpu_count) + ' GPUs.')
            print('Number of epochs        : ' + str(self.config.trainer.num_epochs))
            if type(self.x_train) is not list:
                print('X training data         : ', self.x_train.shape)
            if type(self.y_train) is not list:
                print('Y training data         : ', self.y_train.shape)
            if type(self.x_test) is not list:
                print('X test data             : ', self.x_test.shape)
            if type(self.y_test) is not list:
                print('Y test data             : ', self.y_test.shape)
            if self.config.callbacks.modelcheckpoint:
                print('Callbacks               : Model Checkpoint')
            if self.config.callbacks.earlystopping:
                print('                        : Early Stopping')
            if self.config.callbacks.tensorboard:
                print('                        : Tensor Board')
            if self.config.model.optimizer == 'SGD' and \
                    self.config.callbacks.reduceLR:
                print('                        : Reduce Learning Rate On Plateau')
            if self.config.callbacks.earlystopping:
                print('Patiance of ES          : ' + str(self.config.callbacks.es_patience))
            if self.config.model.optimizer == 'SGD' and \
                    self.config.callbacks.reduceLR:
                print('Patiance of RLRP        : ' + str(self.config.callbacks.reduceLR_patience))
            print('\n---------------------------------------------------\n')

            print('Training is now begining.')

            start = time.time()
            history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.config.trainer.num_epochs,
                batch_size=self.config.trainer.batch_size,
                shuffle=self.config.trainer.shuffle,
                validation_data=(
                    self.x_test,
                    self.y_test
                ),
                callbacks=self.callbacks,
                verbose=self.config.trainer.verbose
            )
            elapsed_time = time.time() - start
        self.val_loss_time['time'].append(elapsed_time)
        self.val_loss_time['val_loss'].append(
            min(history.history['val_loss'])
        )
        df_results = pd.DataFrame(history.history)
        df_results['epoch'] = history.epoch
        os.makedirs(
            self.config.data_loader.path_to_present_dir +
            self.config.callbacks.save_file +
            'History/',
            exist_ok=True
        )
        if self.config.save.history_save:
            df_results.to_csv(
                path_or_buf=self.config.data_loader.path_to_present_dir +
                self.config.callbacks.save_file +
                'History/' +
                self.config.callbacks.model_name +
                '.csv',
                index=False
            )
            print('History was saved.')

        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        K.clear_session()
        print('The session was cleared.')

        df = pd.DataFrame(self.val_loss_time)
        os.makedirs(
            self.config.data_loader.path_to_present_dir +
            self.config.callbacks.save_file +
            'val_loss_time/',
            exist_ok=True
        )
        if self.config.save.val_time_save:
            df.to_csv(
                path_or_buf=self.config.data_loader.path_to_present_dir +
                self.config.callbacks.save_file +
                'val_loss_time/' +
                self.config.callbacks.model_name +
                '.csv',
                index=None
            )
            print('Validation loss & time were saved.')

class trainer_with_CV():
    def __init__(self, model, data, config):
        self.model_maker = model
        self.data = data
        self.config = config
        self.model = None
        self.val_loss_time = {
            'val_loss': [],
            'time': []
        }
        self.model_number = \
            ['Model_' + str(i + 1) for i in range(self.config.trainer.n_splits)]

    def make_callbacks(self):
        self.callbacks = []
        if self.config.callbacks.modelcheckpoint and \
            not self.config.GPU.multi_gpu:
            os.makedirs(
                self.config.data_loader.path_to_present_dir +
                self.config.callbacks.save_file +
                'Model/' + 
                self.config.callbacks.model_name +
                '/',
                exist_ok=True
            )
            self.callbacks.append(
                ModelCheckpoint(
                    self.config.data_loader.path_to_present_dir +
                    self.config.callbacks.save_file +
                    'Model/' +
                    self.config.callbacks.model_name +
                    '/' +
                    self.model_number[self.i] +
                    '.hdf5',
                    monitor=self.config.callbacks.cp_monitor,
                    save_best_only=self.config.callbacks.cp_save_best_only,
                    verbose=self.config.callbacks.cp_verbose
                )
            )
        if self.config.callbacks.modelcheckpoint and \
            self.config.GPU.multi_gpu:
            os.makedirs(
                self.config.data_loader.path_to_present_dir +
                self.config.callbacks.save_file +
                'Model/' + 
                self.config.callbacks.model_name +
                '/',
                exist_ok=True
            )
            self.callbacks.append(
                MultiGPUCheckpointCallback(
                    self.config.data_loader.path_to_present_dir +
                    self.config.callbacks.save_file +
                    'Model/' +
                    self.config.callbacks.model_name +
                    '/' +
                    self.model_number[self.i] +
                    '.hdf5',
                    self.base_model,
                    monitor=self.config.callbacks.cp_monitor,
                    save_best_only=self.config.callbacks.cp_save_best_only,
                    verbose=self.config.callbacks.cp_verbose
                )
            )
        
        if self.config.callbacks.earlystopping:
            self.callbacks.append(
                EarlyStopping(
                    monitor=self.config.callbacks.es_monitor,
                    patience=self.config.callbacks.es_patience,
                    verbose=self.config.callbacks.es_verbose
                )
            )

        if self.config.callbacks.tensorboard:
            self.callbacks.append(
                TensorBoard(
                    log_dir=self.config.callbacks.tensorboard_log_dir,
                    write_graph=self.config.callbacks.tensorboard_write_graph,
                )
            )

        if self.config.model.optimizer == 'SGD' and \
                self.config.callbacks.reduceLR:
            self.callbacks.append(
                ReduceLROnPlateau(
                    monitor=self.config.callbacks.reduceLR_monitor,
                    factor=self.config.callbacks.reduceLR_factor,
                    patience=self.config.callbacks.reduceLR_patience,
                    verbose=self.config.callbacks.reduceLR_verbose,
                )
            )

    def train(self):
        print('Training is now begining.')
        print('The model will be trained using cross validation')
        print('The number of total fold is ' +
              str(self.config.trainer.n_splits))

        kf = KFold(n_splits=self.config.trainer.n_splits, random_state=None, shuffle=True)
        for self.i, (train_index, test_index) in enumerate(kf.split(self.data.X)):
            if self.config.data_loader.generator:
                self.x_test, self.y_test = \
                    self.data.make_val_data()

                print('\n-----------------Training Condition----------------\n')
                if self.config.GPU.multi_gpu:
                    print('This model will be trained using ' + str(self.config.GPU.gpu_count) + ' GPUs.')
                print('This model will be trained by fit_generator.')
                print('Number of epochs        : ' + str(self.config.trainer.num_epochs))
                if self.config.model.name != 'LSTM' and not self.config.data_loader.with_shape:
                    print('X test data             : ', self.x_test.shape)
                    print('Y test data             : ', self.y_test.shape)
                if self.config.callbacks.modelcheckpoint:
                    print('Callbacks               : Model Checkpoint')
                if self.config.callbacks.earlystopping:
                    print('                        : Early Stopping')
                if self.config.callbacks.tensorboard:
                    print('                        : Tensor Board')
                if self.config.model.optimizer == 'SGD' and \
                    self.config.callbacks.reduceLR:
                    print('                        : Reduce Learning Rate On Plateau')
                if self.config.callbacks.earlystopping:
                    print('Patiance of ES          : ' + str(self.config.callbacks.es_patience))
                if self.config.model.optimizer == 'SGD' and \
                    self.config.callbacks.reduceLR:
                    print('Patiance of RLRP        : ' + str(self.config.callbacks.reduceLR_patience))
                print('\nThe nomber of fold      : %d' % (self.i + 1))
                print('\n---------------------------------------------------\n')

                print('Training is now begining.')

            else:
                if self.config.model.name == 'CNN-AE':
                    self.x_train, self.x_test = \
                        self.data.X[train_index], self.data.X[test_index]
                    self.y_train, self.y_test = \
                        self.data.X[train_index, :, :, :self.config.data_loader.phys_num], \
                            self.data.X[test_index, :, :, :self.config.data_loader.phys_num]
                else:
                    if self.config.data_loader.with_shape:
                        self.x_train, self.x_test = \
                            [self.data.X_CNN[train_index], self.data.X[train_index]], \
                                [self.data.X_CNN[test_index], self.data.X[test_index]]
                        self.y_train, self.y_test = \
                            self.data.Y[train_index], self.data.Y[test_index]
                    else:
                        self.x_train, self.x_test = \
                            self.data.X[train_index], self.data.X[test_index]
                        self.y_train, self.y_test = \
                            self.data.Y[train_index], self.data.Y[test_index]

                print('\n-----------------Training Condition----------------\n')
                if self.config.GPU.multi_gpu:
                    print('This model will be trained using ' + str(self.config.GPU.gpu_count) + ' GPUs.')
                print('Number of epochs        : ' + str(self.config.trainer.num_epochs))
                if self.config.model.name != 'LSTM' and not self.config.data_loader.with_shape:
                    print('X training data         : ', self.x_train.shape)
                    print('Y training data         : ', self.y_train.shape)
                    print('X test data             : ', self.x_test.shape)
                    print('Y test data             : ', self.y_test.shape)
                if self.config.callbacks.modelcheckpoint:
                    print('Callbacks               : Model Checkpoint')
                if self.config.callbacks.earlystopping:
                    print('                        : Early Stopping')
                if self.config.callbacks.tensorboard:
                    print('                        : Tensor Board')
                if self.config.model.optimizer == 'SGD' and \
                    self.config.callbacks.reduceLR:
                    print('                        : Reduce Learning Rate On Plateau')
                if self.config.callbacks.earlystopping:
                    print('Patiance of ES          : ' + str(self.config.callbacks.es_patience))
                if self.config.model.optimizer == 'SGD' and \
                    self.config.callbacks.reduceLR:
                    print('Patiance of RLRP        : ' + str(self.config.callbacks.reduceLR_patience))
                print('\nThe nomber of fold      : %d' % (self.i + 1))
                print('\n---------------------------------------------------\n')
                
            if self.model is not None:
                del self.model
            self.model, self.base_model = self.model_maker.make_model()
            
            self.make_callbacks()

            if self.config.data_loader.generator:
                start = time.time()
                history = self.model.fit_generator(
                    self.data,
                    epochs=self.config.trainer.num_epochs,
                    verbose=self.config.trainer.verbose,
                    callbacks=self.callbacks,
                    validation_data=(
                        self.x_test,
                        self.y_test
                    ),
                    max_queue_size=20,
                    workers=10,
                    use_multiprocessing=True,
                    shuffle=self.config.trainer.shuffle,
                )
                elapsed_time = time.time() - start

            else:
                start = time.time()
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=self.config.trainer.num_epochs,
                    batch_size=self.config.trainer.batch_size,
                    shuffle=self.config.trainer.shuffle,
                    validation_data=(
                        self.x_test,
                        self.y_test
                    ),
                    callbacks=self.callbacks,
                    verbose=self.config.trainer.verbose
                )
                elapsed_time = time.time() - start
            self.val_loss_time['time'].append(elapsed_time)
            self.val_loss_time['val_loss'].append(
                min(history.history['val_loss'])
            )
            df_results = pd.DataFrame(history.history)
            df_results['epoch'] = history.epoch
            os.makedirs(
                self.config.data_loader.path_to_present_dir +
                self.config.callbacks.save_file +
                'History/' +
                self.config.callbacks.model_name +
                '/',
                exist_ok=True
            )
            if self.config.save.history_save:
                df_results.to_csv(
                    path_or_buf=self.config.data_loader.path_to_present_dir +
                    self.config.callbacks.save_file +
                    'History/' +
                    self.config.callbacks.model_name +
                    '/' + 
                    self.model_number[self.i] +
                    '.csv',
                    index=False
                )
                print('History was saved.')

            print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            K.clear_session()
            print('The session was cleared.')

        df = pd.DataFrame(self.val_loss_time, index=self.model_number)
        os.makedirs(
            self.config.data_loader.path_to_present_dir +
            self.config.callbacks.save_file +
            'val_loss_time/',
            exist_ok=True
        )
        if self.config.save.val_time_save:
            df.to_csv(
                path_or_buf=self.config.data_loader.path_to_present_dir +
                self.config.callbacks.save_file +
                'val_loss_time/' +
                self.config.callbacks.model_name +
                '.csv',
            )
            print('Validation loss & time were saved.')