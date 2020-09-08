class Dataloader_stateful_LSTM():
    def __init__(self, config):
        self.config = config
        self.num_of_ts = self.config.data_loader.num_of_ts
        self.sequence = self.config.data_loader.sequence
        self.val_ratio = self.config.data_loader.ratio_tr_te
        self.tra_ratio = 1 - self.val_ratio
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
        self.test_train_split()
    
    def data_load(self):
        self.data_LSTM = pd.read_csv(
            self.path_data, header=None, delim_whitespace=False
            )
        self.data_LSTM = self.data_LSTM.values

        if self.config.data_loader.with_shape:
            self.X_CNN = np.zeros([
                self.number_of_shape, self.num_of_ts,
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
                self.X_CNN[i, :, :, :, 0] = data_CNN

    def make_dataset(self):
        self.X = np.zeros(
            [
                self.number_of_shape,
                self.num_of_ts,
                self.maxlen,
                self.data_size
            ]
        )
        self.Y = np.zeros(
            [
                self.number_of_shape,
                self.num_of_ts,
                self.data_size
            ]
        )
        for i in range(self.number_of_shape):
            for j in range(self.num_of_ts):
                self.X[i, j] = \
                    self.data_LSTM[
                        i * self.num_of_ts_for_data + j : \
                            i * self.num_of_ts_for_data + j + \
                            self.time_step * self.maxlen : self.time_step
                    ]
                self.Y[i, j] = \
                    self.data_LSTM[
                        i * self.num_of_ts_for_data + j + \
                            self.time_step * self.maxlen
                    ]
    
    def generate_training_data(self):
        if self.config.data_loader.with_shape:
            while True:
                order = np.arange(self.num_of_ts * self.tra_ratio - self.sequence + 1, dtype='int')
                random.shuffle(order)
                for i in range(int(self.num_of_ts * self.tra_ratio - self.sequence + 1)):
                    for j in range(self.number_of_shape):
                        X_C, X_L = self.X_train
                        X_CNN = X_C[j, order[i]:order[i] + self.sequence]
                        X_CNN = X_CNN.reshape(self.sequence, 120, 120, 1)
                        X_LSTM = X_L[j, order[i]:order[i] + self.sequence]
                        X_LSTM = X_LSTM.reshape(self.sequence, self.maxlen, self.data_size)
                        Y = self.y_train[j, order[i]:order[i] + self.sequence]
                        Y = Y.reshape(self.sequence, self.data_size)
                        yield [X_CNN, X_LSTM], Y
        else:
            while True:
                order = np.arange(self.num_of_ts * self.tra_ratio - self.sequence + 1, dtype='int')
                random.shuffle(order)
                for i in range(int(self.num_of_ts * self.tra_ratio - self.sequence + 1)):
                    for j in range(self.number_of_shape):
                        X = self.X_train[j, order[i]:order[i] + self.sequence]
                        X = X.reshape(self.sequence, self.maxlen, self.data_size)
                        Y = self.y_train[j, order[i]:order[i] + self.sequence]
                        Y = Y.reshape(self.sequence, self.data_size)
                        yield X, Y

    def generate_validation_data(self):
        if self.config.data_loader.with_shape:
            while True:
                for i in range(self.number_of_shape):
                    X_C, X_L = self.X_test
                    X_CNN = X_C[i]
                    X_CNN = X_CNN.reshape(-1, 120, 120, 1)
                    X_LSTM = X_L[i]
                    X_LSTM = X_LSTM.reshape(-1, self.maxlen, self.data_size)
                    Y = self.y_test[i]
                    Y = Y.reshape(-1, self.data_size)
                    yield [X_CNN, X_LSTM], Y
        else:
            while True:
                for i in range(self.number_of_shape):
                    X = self.X_test[i]
                    X = X.reshape(-1, self.maxlen, self.data_size)
                    Y = self.y_test[i]
                    Y = Y.reshape(-1, self.data_size)
                    yield X, Y

    def test_train_split(self):
        if self.config.data_loader.with_shape:
            len_val = int(self.num_of_ts * self.config.data_loader.ratio_tr_te)
            X_CNN_train, X_CNN_test, \
                X_train, X_test, \
                self.y_train, self.y_test = \
                self.X_CNN[:, len_val:], self.X_CNN[:, :len_val], \
                self.X[:, len_val:], self.X[:, :len_val], \
                self.Y[:, len_val:], self.Y[:, :len_val]
            self.X_train = [X_CNN_train, X_train]
            self.X_test = [X_CNN_test, X_test]

        else:
            len_val = int(self.num_of_ts * self.config.data_loader.ratio_tr_te)
            self.X_train, self.X_test, \
                self.y_train, self.y_test = \
                self.X[:, len_val:], self.X[:, :len_val], \
                self.Y[:, len_val:], self.Y[:, :len_val]