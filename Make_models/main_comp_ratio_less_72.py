from dataloaders.Dataloader import (Dataloader, Dataloader_LSTM,
                                    Datagenerator_for_AE, gen_for_AE_with_shape,
                                    Datagenerator_for_AE_with_CV)
from models.CNN_AE_2 import MS_CNN_AE
from models.LSTM import simple_LSTM, LSTM_with_shape
from utils.config import process_config
from utils.utils import get_args
from trainers.Trainer import (trainer, trainer_with_CV)

import pandas as pd
from tqdm import tqdm

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    if config.GPU.multi_gpu:
        config.trainer.batch_size = config.trainer.batch_size * config.GPU.gpu_count

    VAL_TIME = {
        'val_loss':[],
        'time':[]
    }

    params_1 = [2, 1]
    params_2 = [False, True]
    modelname = ['6_3_2', '2_1_1']
    folder_name = 'comp_ratio/'
    
    for i in tqdm(range(len(params_1))):
        config.model.convert_part_chanel_nm = params_1[i]
        config.model.less_than_72 = params_2[i]
        config.callbacks.model_name = folder_name + modelname[i]


        if i == 0:
            print('Create the data generator.')
            if config.model.name == 'CNN-AE':
                if config.data_loader.generator:
                    if config.data_loader.with_shape:
                        data_loader = gen_for_AE_with_shape(config)
                    else:
                        if config.trainer.use_CV:
                            data_loader = Datagenerator_for_AE_with_CV(config)
                        else:
                            data_loader = Datagenerator_for_AE(config)
                else:
                    data_loader = Dataloader(config)
            elif config.model.name == 'LSTM':
                data_loader = Dataloader_LSTM(config)
        

        print('Create the model.')
        if config.model.name == 'CNN-AE':
            if config.model.multi_scale:
                model = MS_CNN_AE(config)
        elif config.model.name == 'LSTM':
            if config.data_loader.with_shape:
                model = LSTM_with_shape(config)
            else:
                model = simple_LSTM(config)

        if config.trainer.use_CV:
            TRAINER = trainer_with_CV(model, data_loader, config)
        else:
            TRAINER = trainer(model, data_loader, config)
        print('Trainer was created.\n')

        TRAINER.train()
        VAL_TIME['val_loss'].append(TRAINER.val_loss_time['val_loss'])
        VAL_TIME['time'].append(TRAINER.val_loss_time['time'])

    df = pd.DataFrame(VAL_TIME, index=modelname)
        
    df.to_csv(
        path_or_buf=config.data_loader.path_to_present_dir +
        config.callbacks.save_file +
        'val_loss_time/comp_ratio_less_72.csv',
        )

if __name__ == '__main__':
    main()

