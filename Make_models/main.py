from dataloaders.Dataloader import *
from models.CNN_AE import *
from models.LSTM import *
from utils.config import process_config
from utils.utils import get_args
from trainers.Trainer import *

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    
    if config.GPU.multi_gpu:
        config.trainer.batch_size = config.trainer.batch_size * config.GPU.gpu_count

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
        if config.data_loader.generator:
            data_loader = gen_for_LSTM(config)
        else:
            data_loader = Dataloader_LSTM(config)

    print('Create the model.')
    if config.model.name == 'CNN-AE':
        if config.model.multi_scale and not config.model.fc:
            model = MS_CNN_AE(config)
        elif config.model.multi_scale and config.model.fc:
            model = MS_FC_CNN_AE(config)
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


if __name__ == '__main__':
    main()
