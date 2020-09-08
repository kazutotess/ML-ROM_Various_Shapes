import numpy as np
import math
from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, KFold
from keras.utils import Sequence
import random


class Dataloader():
    def __init__(self, config):
        self.config = config
        self.num_of_ts = self.config.data_loader.num_of_ts
        self.path_to_present_dir = self.config.data_loader.path_to_present_dir
        self.x_num = self.config.data_loader.x_num
        self.y_num = self.config.data_loader.y_num
        self.phys_num = self.config.data_loader.phys_num
        self.kind_num = self.config.data_loader.kind_num
        self.X = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.data_load()

    def data_load(self):
        if self.config.data_loader.train:
            self.X = np.zeros([self.kind_num, self.num_of_ts,
                               self.x_num, self.y_num, self.phys_num])
            if self.config.data_loader.with_shape:
                self.X = np.zeros([self.kind_num, self.num_of_ts,
                                   self.x_num, self.y_num, self.phys_num + 1])

            for i in tqdm(range(1, self.kind_num + 1)):
                fnstr = self.path_to_present_dir + '/pickles/data_' + \
                    '{0:03d}'.format(i)+'.pickle'

                if self.config.data_loader.with_shape:
                    tempfn = self.path_to_present_dir + \
                        '/CNN_autoencoder/Flags_for_training/Flag' + \
                        '{0:03d}'.format(i)+'.csv'
                    data = pd.read_csv(tempfn, header=None,
                                       delim_whitespace=False)
                    data = data.values
                # Pickle load

                with open(fnstr, 'rb') as f:
                    obj = pickle.load(f)
                self.X[i - 1, :, :, :, :self.phys_num] = obj[:self.num_of_ts]

                if self.config.data_loader.with_shape:
                    self.X[i - 1, :, :, :, self.phys_num] = data

            self.X = np.reshape(
                self.X,
                [-1,
                 self.X.shape[-3],
                 self.X.shape[-2],
                 self.X.shape[-1]]
            )

        else:
            self.X = np.zeros([self.aumont_kind - self.kind_num,
                               self.num_of_ts, self.x_num, self.y_num,
                               self.phys_num])

            for i in tqdm(
                range(1, self.config.aumont_kind - self.kind_num + 1)
            ):
                fnstr = self.path_to_present_dir + '/pickles/Test_data/data_' \
                    + '{0:03d}'.format(i)+'.pickle'

                if self.config.data_loader.with_shape:
                    tempfn = self.path_to_present_dir + \
                        '/CNN_autoencoder/Flags_for_training/Test_data/Flag' \
                        + '{0:03d}'.format(i)+'.csv'
                    data = pd.read_csv(tempfn, header=None,
                                       delim_whitespace=False)
                    data = data.values

                # Pickle load
                with open(fnstr, 'rb') as f:
                    obj = pickle.load(f)
                self.X[i - 1, :, :, :, :self.phys_num] = \
                    obj[:self.x_numnum_of_ts]

                if self.config.data_loader.with_shape:
                    self.X[i - 1, :, :, :, self.phys_num] = data

    def test_train_split(self):
        x_train, x_test, y_train, y_test = \
            train_test_split(self.X,
                             self.X[:, :, :, :self.phys_num],
                             test_size=self.config.data_loader.ratio_tr_te,
                             random_state=None)
        return x_train, x_test, y_train, y_test

class Dataloader_LSTM():
    def __init__(self, config):
        self.config = config
        self.num_of_ts = self.config.data_loader.num_of_ts
        self.num_of_ts_for_data = self.config.data_loader.num_of_ts_for_data
        self.number_of_shape = self.config.data_loader.kind_num
        self.maxlen = self.config.data_loader.maxlen
        self.time_step = self.config.data_loader.time_step
        self.data_size = self.config.data_loader.data_size
        self.path_to_present_dir = self.config.data_loader.path_to_present_dir
        self.path_data = self.path_to_present_dir + '/LSTM/Dataset/' + \
            self.config.data_loader.dataset_name
            
        assert self.num_of_ts + self.time_step * (self.maxlen - 1) < \
            self.num_of_ts_for_data, 'The data aumont is not enough.'

        self.data_load()
        self.make_dataset()
    
    def data_load(self):
        self.data_LSTM = pd.read_csv(
            self.path_data, header=None, delim_whitespace=False
            )
        self.data_LSTM = self.data_LSTM.values

        if self.config.data_loader.with_shape:
            self.X_CNN = np.zeros([
                self.number_of_shape * self.num_of_ts,
                120, 120, 1
            ])
            for i in range(self.number_of_shape):
                data_CNN = pd.read_csv(
                    self.path_to_present_dir + \
                        '/LSTM/Flags_for_training_LSTM/Flag' + \
                        '{0:03d}'.format(i + 1) + '.csv',
                    header=None,
                    delim_whitespace=False
                )
                data_CNN = data_CNN.values
                self.X_CNN[
                    i * self.num_of_ts : (i + 1) * self.num_of_ts,
                    :, :, 0
                    ] = data_CNN

    def make_dataset(self):
        self.X = np.zeros(
            [
                self.number_of_shape * self.num_of_ts,
                self.maxlen,
                self.data_size
            ]
        )
        if self.config.model.return_sequence[-1]:
            self.Y = np.zeros(
                [
                    self.number_of_shape * self.num_of_ts,
                    self.maxlen,
                    self.data_size
                ]
            )
        else:
            self.Y = np.zeros(
                [
                    self.number_of_shape * self.num_of_ts,
                    self.data_size
                ]
            )
        for i in range(self.number_of_shape):
            for j in range(self.num_of_ts):
                self.X[i * self.num_of_ts + j] = \
                    self.data_LSTM[
                        i * self.num_of_ts_for_data + j : \
                            i * self.num_of_ts_for_data + j + \
                            self.time_step * self.maxlen : self.time_step
                    ]
                if self.config.model.return_sequence[-1]:
                    self.Y[i * self.num_of_ts + j] = \
                        self.data_LSTM[
                            i * self.num_of_ts_for_data + j + 1 : \
                                i * self.num_of_ts_for_data + j + \
                                self.time_step * self.maxlen + 1: self.time_step
                        ]
                else:
                    self.Y[i * self.num_of_ts + j] = \
                        self.data_LSTM[
                            i * self.num_of_ts_for_data + j + \
                                self.time_step * self.maxlen
                        ]

    def test_train_split(self):
        if self.config.data_loader.with_shape:
            X_CNN_train, X_CNN_test, X_train, X_test, y_train, y_test = \
                train_test_split(self.X_CNN,
                                self.X,
                                self.Y,
                                test_size=self.config.data_loader.ratio_tr_te,
                                random_state=None)
            x_train = [X_CNN_train, X_train]
            x_test = [X_CNN_test, X_test]
            return x_train, x_test, y_train, y_test

        else:
            x_train, x_test, y_train, y_test = \
                train_test_split(self.X,
                                self.Y,
                                test_size=self.config.data_loader.ratio_tr_te,
                                random_state=None)
            return x_train, x_test, y_train, y_test

class Datagenerator_for_AE(Sequence):
    def __init__(self, config):
        self.config = config
        self.path_to_present_dir = self.config.data_loader.path_to_present_dir
        self.batch_size = self.config.trainer.batch_size
        self.x_num = self.config.data_loader.x_num
        self.y_num = self.config.data_loader.y_num
        self.phys_num = self.config.data_loader.phys_num
        self.index = []
        self.devide_train_test()
        self.length = math.ceil(len(self.X) / self.n_split * (self.n_split - 1) / self.batch_size)
    
    def devide_train_test(self):
        for i in range(self.config.data_loader.kind_num):
            for j in range(self.config.data_loader.num_of_ts):
                self.index.append([i + 1,  j + 1])
        random.shuffle(self.index)
        self.val_num = int(len(self.index) * self.config.data_loader.ratio_tr_te)
        self.val_index = self.index[-self.val_num:]
        self.tra_index = self.index[:-self.val_num]

    def make_val_data(self):
        val_data = np.zeros([len(self.val_index), self.x_num, 
                            self.y_num, 
                            self.phys_num])
        print('\nLoading the validation data.')
        for i in tqdm(range(len(self.val_index))):
            file_name = self.path_to_present_dir + '/Training_data/UVP_' + \
                '{0:03d}'.format(self.val_index[i][0]) + '_' + '{0:04d}'.format(self.val_index[i][1])
            VAL_DATA = pd.read_csv(file_name, header=None, delim_whitespace=False)
            VAL_DATA = np.array(VAL_DATA)
            val_data[i] = VAL_DATA.reshape(
                            self.x_num, 
                            self.y_num, 
                            self.phys_num)
        return val_data, val_data

    def __getitem__(self, idx):
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > len(self.tra_index):
            bs = len(self.tra_index) - start_pos
        else:
            bs = self.batch_size
        tra_data = np.zeros([bs, self.x_num, self.y_num, self.phys_num])
        for i in range(bs):
            file_name = self.path_to_present_dir + '/Training_data/UVP_' + \
                    '{0:03d}'.format(self.tra_index[idx * self.batch_size + i][0]) + '_' + \
                        '{0:04d}'.format(self.tra_index[idx * self.batch_size + i][1])
            TRA_DATA = pd.read_csv(file_name, header=None, delim_whitespace=False)
            TRA_DATA = np.array(TRA_DATA)
            tra_data[i] = TRA_DATA.reshape(self.x_num, 
                                self.y_num, 
                                self.phys_num)
        return tra_data, tra_data

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        pass

class gen_for_AE_with_shape(Sequence):
    def __init__(self, config):
        self.config = config
        self.path_to_present_dir = self.config.data_loader.path_to_present_dir
        self.batch_size = self.config.trainer.batch_size
        self.x_num = self.config.data_loader.x_num
        self.y_num = self.config.data_loader.y_num
        self.phys_num = self.config.data_loader.phys_num
        self.index = []
        self.devide_train_test()
        self.length = math.ceil(len(self.X) / self.n_split * (self.n_split - 1) / self.batch_size)
    
    def devide_train_test(self):
        for i in range(self.config.data_loader.kind_num):
            for j in range(self.config.data_loader.num_of_ts):
                self.index.append([i + 1,  j + 1])
        random.shuffle(self.index)
        self.val_num = int(len(self.index) * self.config.data_loader.ratio_tr_te)
        self.val_index = self.index[-self.val_num:]
        self.tra_index = self.index[:-self.val_num]

    def make_val_data(self):
        val_data = np.zeros([len(self.val_index),
                            self.x_num, self.y_num, self.phys_num + 1])
        print('\nLoading the validation data.')
        for i in tqdm(range(len(self.val_index))):
            file_name = self.path_to_present_dir + '/Training_data/UVP_' + \
                '{0:03d}'.format(self.val_index[i][0]) + '_' + '{0:04d}'.format(self.val_index[i][1])
            VAL_DATA = pd.read_csv(file_name, header=None, delim_whitespace=False)
            VAL_DATA = np.array(VAL_DATA)
            val_data[i, :, :, :self.phys_num] = VAL_DATA.reshape(
                            self.x_num, 
                            self.y_num, 
                            self.phys_num)
            tempfn = self.path_to_present_dir + \
                '/CNN_autoencoder/Flags_for_training/Flag' + \
                '{0:03d}'.format(self.val_index[i][0])+'.csv'
            data = pd.read_csv(tempfn, header=None,
                                delim_whitespace=False)
            data = data.values
            val_data[i, :, :, self.phys_num] = data
        return val_data, val_data[:, :, :, :self.phys_num]

    def __getitem__(self, idx):
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > len(self.tra_index):
            bs = len(self.tra_index) - start_pos
        else:
            bs = self.batch_size
        tra_data = np.zeros([bs,
                            self.x_num, self.y_num, self.phys_num + 1])
        for i in range(bs):
            file_name = self.path_to_present_dir + '/Training_data/UVP_' + \
                    '{0:03d}'.format(self.tra_index[idx * self.batch_size + i][0]) + '_' + \
                        '{0:04d}'.format(self.tra_index[idx * self.batch_size + i][1])
            TRA_DATA = pd.read_csv(file_name, header=None, delim_whitespace=False)
            TRA_DATA = np.array(TRA_DATA)
            tra_data[i, :, :, :self.phys_num] = TRA_DATA.reshape(self.x_num, 
                                self.y_num, 
                                self.phys_num)
            tempfn = self.path_to_present_dir + \
                '/CNN_autoencoder/Flags_for_training/Flag' + \
                '{0:03d}'.format(self.tra_index[idx * self.batch_size + i][0])+'.csv'
            data = pd.read_csv(tempfn, header=None,
                                delim_whitespace=False)
            data = data.values
            tra_data[i, :, :, self.phys_num] = data
        return tra_data, tra_data[:, :, :, :self.phys_num]

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        pass

class Datagenerator_for_AE_with_CV(Sequence):
    def __init__(self, config):
        self.config = config
        self.n_split = self.config.trainer.n_splits
        self.path_to_present_dir = self.config.data_loader.path_to_present_dir
        self.batch_size = self.config.trainer.batch_size
        self.x_num = self.config.data_loader.x_num
        self.y_num = self.config.data_loader.y_num
        self.phys_num = self.config.data_loader.phys_num
        self.X = []
        for i in range(self.config.data_loader.kind_num):
            for j in range(self.config.data_loader.num_of_ts):
                self.X.append([i + 1,  j + 1])
        self.X = np.array(self.X)
        self.index_generator = self.devide_train_test()
        self.length = math.ceil(len(self.X) / self.n_split * (self.n_split - 1) / self.batch_size)
    
    def devide_train_test(self):
        self.val_num = int(len(self.X) * self.config.data_loader.ratio_tr_te)
        kf = KFold(n_splits=self.n_split, random_state=None, shuffle=True)
        while True:
            for train_index, test_index in kf.split(self.X):
                self.tra_index, self.val_index = \
                        self.X[train_index], self.X[test_index]
                yield

    def make_val_data(self):
        next(self.index_generator)
        val_data = np.zeros([len(self.val_index), self.x_num, 
                            self.y_num, 
                            self.phys_num])
        print('\nLoading the validation data.')
        for i in tqdm(range(len(self.val_index))):
            file_name = self.path_to_present_dir + '/Training_data/UVP_' + \
                '{0:03d}'.format(self.val_index[i][0]) + '_' + '{0:04d}'.format(self.val_index[i][1])
            VAL_DATA = pd.read_csv(file_name, header=None, delim_whitespace=False)
            VAL_DATA = np.array(VAL_DATA)
            val_data[i] = VAL_DATA.reshape(
                            self.x_num, 
                            self.y_num, 
                            self.phys_num)
        return val_data, val_data

    def __getitem__(self, idx):
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > len(self.tra_index):
            bs = len(self.tra_index) - start_pos
        else:
            bs = self.batch_size
        tra_data = np.zeros([bs, self.x_num, self.y_num, self.phys_num])
        for i in range(bs):
            file_name = self.path_to_present_dir + '/Training_data/UVP_' + \
                    '{0:03d}'.format(self.tra_index[idx * self.batch_size + i][0]) + '_' + \
                        '{0:04d}'.format(self.tra_index[idx * self.batch_size + i][1])
            TRA_DATA = pd.read_csv(file_name, header=None, delim_whitespace=False)
            TRA_DATA = np.array(TRA_DATA)
            tra_data[i] = TRA_DATA.reshape(self.x_num, 
                                self.y_num, 
                                self.phys_num)
        return tra_data, tra_data

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        pass

class gen_for_LSTM(Sequence):
    def __init__(self, config):
        self.config = config
        self.num_of_ts = self.config.data_loader.num_of_ts
        self.num_of_ts_for_data = self.config.data_loader.num_of_ts_for_data
        self.number_of_shape = self.config.data_loader.kind_num
        self.maxlen = self.config.data_loader.maxlen
        self.time_step = self.config.data_loader.time_step
        self.data_size = self.config.data_loader.data_size
        self.val_ratio = self.config.data_loader.ratio_tr_te
        self.tra_ratio = 1 - self.val_ratio
        self.len_val = int(self.num_of_ts * self.val_ratio)
        self.len_tra = int(self.num_of_ts * self.tra_ratio)
        self.path_to_present_dir = self.config.data_loader.path_to_present_dir
        self.path_data = self.path_to_present_dir + '/LSTM/Dataset/' + \
            self.config.data_loader.dataset_name
        self.data_load()
        self.make_dataset()
    
    def data_load(self):
        self.data_LSTM = pd.read_csv(
            self.path_data, header=None, delim_whitespace=False
            )
        self.data_LSTM = self.data_LSTM.values
        self.data_LSTM = np.reshape(
            self.data_LSTM, (
                80,
                self.num_of_ts_for_data,
                self.data_size
            )
        )

        if self.config.data_loader.with_shape:
            self.data_CNN = np.zeros([
                self.number_of_shape, self.num_of_ts,
                120, 120, 1
            ])
            for i in range(self.number_of_shape):
                DATA_CNN = pd.read_csv(
                    self.path_to_present_dir + \
                        '/LSTM/Flags_for_training_LSTM/Flag' + \
                        '{0:03d}'.format(i + 1) + '.csv',
                    header=None,
                    delim_whitespace=False
                )
                DATA_CNN = DATA_CNN.values
                self.data_CNN[i, :, :, :, 0] = DATA_CNN

    def make_dataset(self):
        self.X_LSTM = []
        if self.config.data_loader.with_shape:
            self.X_CNN = []
        self.Y = []
        self.length = []
        for i in range(self.len_tra):
            LENGTH = int(random.uniform(1, self.maxlen))
            self.length.append(LENGTH)
            input_LSTM = self.data_LSTM[
                :self.number_of_shape, i : i + self.time_step * LENGTH : self.time_step
            ]
            self.X_LSTM.append(input_LSTM)
            if self.config.data_loader.with_shape:
                input_CNN = self.data_CNN[:, i]
                self.X_CNN.append(input_CNN)
            if self.config.model.return_sequence[-1]:
                output = self.data_LSTM[
                    :self.number_of_shape, i + 1 : i + self.time_step * LENGTH + 1 : self.time_step
                ]
            else:
                output = self.data_LSTM[
                    :self.number_of_shape, i + self.time_step * LENGTH + 1
                ]
            self.Y.append(output)

        self.X_LSTM_VAL = np.zeros([
            self.len_val * self.number_of_shape, self.maxlen, self.data_size
        ])
        if self.config.data_loader.with_shape:
            self.X_CNN_VAL = np.zeros([
                self.len_val * self.number_of_shape, 120, 120, 1
            ])
        if self.config.model.return_sequence[-1]:
            self.Y_VAL = np.zeros([
                self.len_val * self.number_of_shape, self.maxlen, self.data_size
            ])
        else:
            self.Y_VAL = np.zeros([
                self.len_val * self.number_of_shape, 1, self.data_size
            ])
        for i in range(self.len_tra, int(self.len_val + self.len_tra)):
            for j in range(self.number_of_shape):
                self.X_LSTM_VAL[j * self.number_of_shape + i] = \
                    self.data_LSTM[j, i : i + self.time_step * self.maxlen : self.time_step]
                if self.config.data_loader.with_shape:
                    self.X_CNN_VAL[j * self.number_of_shape + i] = \
                        self.data_CNN[j, i]
                if self.config.model.return_sequence[-1]:
                    self.Y_VAL[j * self.number_of_shape + i] = \
                        self.data_LSTM[j,
                        i + 1 : i + self.time_step * self.maxlen + 1 : self.time_step]
                else:
                    self.Y_VAL[j * self.number_of_shape + i] = \
                        self.data_LSTM[j,
                        i + self.time_step * self.maxlen + 1]

    def make_val_data(self):
        if self.config.data_loader.with_shape:
            return [self.X_CNN_VAL, self.X_LSTM_VAL], self.Y_VAL
        else:
            return self.X_LSTM_VAL, self.Y_VAL

    def __getitem__(self, idx):
        if self.config.data_loader.with_shape:
            return [self.X_CNN[idx], self.X_LSTM[idx]], self.Y[idx]
        else:
            return self.X_LSTM[idx], self.Y[idx]

    def __len__(self):
        return self.len_tra

    def on_epoch_end(self):
        pass