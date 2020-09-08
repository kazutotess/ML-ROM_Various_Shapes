class trainer_with_generator():
    def __init__(self, model, generator, config):
        self.model = model.make_model()
        self.generator = generator
        self.config = config
        self.callbacks = []
        self.val_loss_time = {
            'val_loss': [],
            'time': []
        }
        self.init_callbacks()

    def init_callbacks(self):
        if self.config.callbacks.modelcheckpoint:
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
        self.x_test, self.y_test = \
            self.generator.make_val_data()

        print('\n-----------------Training Condition----------------\n')
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
            self.generator,
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

class trainer_with_generator_CV():
    def __init__(self, model, generator, config):
        self.model_maker = model
        self.generator = generator
        self.config = config
        self.model_number = \
            ['Model_' + str(i + 1) for i in range(self.config.trainer.n_splits)]
        self.val_loss_time = {
            'val_loss': [],
            'time': []
        }

    def make_callbacks(self):
        self.callbacks = []
        if self.config.callbacks.modelcheckpoint:
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
        for self.i in range(self.config.trainer.n_splits):
            self.x_test, self.y_test = \
                self.generator.make_val_data()

            print('\n-----------------Training Condition----------------\n')
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

            self.make_callbacks()

            if self.model is not None:
                del self.model
            self.model = self.model_maker.make_model()

            start = time.time()
            history = self.model.fit_generator(
                self.generator,
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

class trainer_for_stateful():
    def __init__(self, model, data, config):
        self.model, self.base_model = model.make_model()
        self.data = data
        self.config = config
        self.num_of_ts = self.config.data_loader.num_of_ts
        self.sequence = self.config.data_loader.sequence
        self.val_ratio = self.config.data_loader.ratio_tr_te
        self.tra_ratio = 1 - self.val_ratio
        self.val_loss_time = {
            'val_loss': [],
            'time': []
        }
        if self.config.callbacks.modelcheckpoint:
            os.makedirs(
                    self.config.data_loader.path_to_present_dir +
                    self.config.callbacks.save_file +
                    'Model/' + 
                    self.config.callbacks.model_name +
                    '/',
                    exist_ok=True
                )
            self.save_file = self.config.data_loader.path_to_present_dir + \
                        self.config.callbacks.save_file + \
                        'Model/' + \
                        self.config.callbacks.model_name + \
                        '.hdf5'

    
    def train(self):
        validation_generator = self.data.generate_validation_data()
        training_generator = self.data.generate_training_data()
        HISTORY = {'loss': []}
        validation_losses = []
        epochs = []
        num_epochs = self.config.trainer.num_epochs
        width = 30
        _total_width = 0
        smallest_loss = float('inf')
        target = int(self.num_of_ts * self.tra_ratio - self.sequence + 1)
        print('\n-----------------Training Condition----------------\n')
        print('This model will be trained by stateful training.')
        if self.config.GPU.multi_gpu:
            print('This model will be trained using ' + str(self.config.GPU.gpu_count) + ' GPUs.')
        print('Number of epochs        : ' + str(self.config.trainer.num_epochs))
        print('\n---------------------------------------------------\n')

        print('Training is now begining.')

        print('\nTrain on %d sequences and %d kinds of shapes' %
            (int(self.num_of_ts * self.tra_ratio - self.sequence + 1),
            self.config.data_loader.kind_num))
        START_TIME = time.time()
        for i in range(num_epochs):
            start = time.time()
            print('\nEpoch ' + str(i + 1) + '/' + str(num_epochs))
            epoch = i + 1
            epochs.append(int(epoch))
            sum_validation = 0
            sum_train = 0

            # Training
            for j in range(target):
                j = j + 1
                prev_total_width = _total_width
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
                numdigits = int(np.floor(np.log10(target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, target)
                bar = barstr % j
                prog = float(j) / target
                prog_width = int(width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if j < target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (width - prog_width))
                bar += ']'
                _total_width = len(bar)
                sys.stdout.write(bar)
                sys.stdout.flush()
                for k in range(self.config.data_loader.kind_num):
                    x_train, y_train = next(training_generator)
                    history = self.model.fit(
                        x_train,
                        y_train,
                        epochs=1,
                        batch_size=1,
                        shuffle=False,
                        verbose=0
                    )
                    sum_train += history.history['loss'][0]
                    self.model.reset_states()
            training_loss = sum_train / (
                int(self.num_of_ts * self.tra_ratio - self.sequence + 1 *
                    self.config.data_loader.kind_num)
            )
            HISTORY['loss'].append(training_loss)

            # Evaluate
            for j in range(self.config.data_loader.kind_num):
                x_test, y_test = next(validation_generator)
                sum_validation += self.model.evaluate(x_test, y_test, batch_size=1, verbose=0)
                self.model.reset_states()
            validation_loss = sum_validation / self.config.data_loader.kind_num
            validation_losses.append(validation_loss)
            elapsed_time = time.time() - start
            sys.stdout.write(' - ' + str(int(elapsed_time))
                            + 's' + ' - loss: %f - val_loss: %f' %(training_loss, validation_loss))
            sys.stdout.flush()
            if self.config.callbacks.modelcheckpoint:
                if smallest_loss > validation_losses[-1]:
                    sys.stdout.write(
                        '\nEpoch ' + '{0:05d}'.format(i + 1) +
                            ': val_loss improved from ' + 
                            '{:.5g}'.format(smallest_loss) + 
                            ' to ' + 
                            '{:.5g}'.format(validation_losses[-1]) +
                            ', saving model to ' + 
                            self.save_file
                    )
                    sys.stdout.flush()
                    smallest_loss = validation_losses[-1]
                    self.model.save(self.save_file)
                else:
                    sys.stdout.write(
                        '\nEpoch ' + '{0:05d}'.format(i + 1) +
                            ': val_loss did not improve'
                    )
                    sys.stdout.flush()
        END_TIME = time.time() - START_TIME
        self.val_loss_time['time'].append(END_TIME)
        self.val_loss_time['val_loss'].append(
            min(validation_losses)
        )
        df_results = pd.DataFrame(HISTORY)
        df_results['val_loss'] = validation_losses
        df_results['epoch'] = epochs
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

        print("elapsed_time:{0}".format(END_TIME) + "[sec]")

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