import dataset
import torch
import glob
import os
from models.cnn import CNN
from config import args, print_args
from trainer import Trainer

if __name__ == '__main__':
    if not args.debug_mode:
        import wandb
        wandb.init(project=args.project, name=args.name, tags=args.tags, config=args)
        train_data = dataset.MDB_Dataset('MusicDelta_80sRock')
        test_data = dataset.MDB_Dataset('MusicDelta_80sRock')
    else:
        train_data = dataset.MDB_Dataset('MusicDelta_80sRock')
        test_data = dataset.MDB_Dataset('MusicDelta_80sRock')        

    print_args(args)

    # get_model
    if args.model_arc == 'CNN':
        model = CNN(hidden_channel_num=10, output_number=4)
    else:
        raise AssertionError

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.debug_mode:
        wandb.watch(model)

    trainer = Trainer(model, optimizer, args.device, args.debug_mode, args.test_per_epoch, args.num_epochs, args.weight_path,
                        train_data, test_data)
    trainer.train()