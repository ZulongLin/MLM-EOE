import shutil
import sys
import os
import argparse
from itertools import product

import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
import lightning as pl
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.loggers import CSVLogger
from dataset import VideoRegressionDataModule
from train_process import VideoRegressionModel
import datetime


class MyTQDMProgressBar(TQDMProgressBar):
    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()

    def init_validation_tqdm(self):
        bar = Tqdm(
            desc=self.validation_description,
            position=0,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--data_dir', default='', type=str,
                        help='data_dir')
    parser.add_argument('--label_file', default='', type=str,
                        help='data_dir')
    parser.add_argument('--train_data', default=['AVEC2013-train'], nargs='+', help='traindata')
    parser.add_argument('--val_data', default=['AVEC2013-dev'], nargs='+', help='valdata')
    parser.add_argument('--test_data', default=['AVEC2013-test'], nargs='+', help='testdata')
    parser.add_argument('--pretrained_way', default='imagenet', type=str, help='pretrained_data')
    parser.add_argument('--cache_path', default=f'AVEC2013', type=str, help='cache_path')
    parser.add_argument('--type', default='AVEC2013', type=str, help='type')
    parser.add_argument('--log_dir', default='', type=str, help='log_dir')
    parser.add_argument("--deviceid", nargs="+", default=[0], type=int, help="A list of integers")
    parser.add_argument('--pretrained', default=False, type=bool, help='num_frames')
    parser.add_argument('--num_frames', default=1, type=int, help='num_frames')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
    parser.add_argument('--max_epochs', default=50, type=int, help='max_epochs')
    parser.add_argument('--frame_interval', default=1, type=int, help='frame_interval')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight_decay')
    parser.add_argument('--dropout', default=0.7, type=float, help='dropout')
    parser.add_argument('--seed', default=15, type=int, help='random_seed')
    parser.add_argument('--search', default=True, type=bool, help='Enable hyperparameter search')
    args = parser.parse_args()


    def seed_torch(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


    frame_interval_values = [5]
    wd_values = [1e-4, 1e-3, 1e-5]
    dropout_values = [0.2, 0.4, 0.7]

    param_combinations = list(product(frame_interval_values, wd_values, dropout_values))

    if args.search:
        for frame_interval, wd, dropout in param_combinations:
            args.frame_interval = frame_interval
            args.wd = wd
            args.dropout = dropout

            log_dir = f'logs/'
            best_model_dir = f'best_models/'

            shutil.rmtree(args.cache_path, ignore_errors=True)
            os.makedirs(args.cache_path, exist_ok=True)

            seed_torch(args.seed)

            data_module = VideoRegressionDataModule(data_dir=args.data_dir,
                                                    label_file=args.label_file,
                                                    num_frames=args.num_frames,
                                                    train_data=args.train_data,
                                                    val_data=args.val_data,
                                                    test_data=args.test_data,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    frame_interval=args.frame_interval,
                                                    cache_path=args.cache_path,
                                                    type=args.type)
            model = VideoRegressionModel(args)

            checkpoint_callback = ModelCheckpoint(
                monitor='val_mae',
                mode='min',
                dirpath=best_model_dir,
                filename='{epoch}-{val_mae:.2f}-{val_rmse:.2f}'
            )

            trainer = pl.Trainer(
                accelerator="gpu",
                devices=args.deviceid,
                logger=CSVLogger(save_dir=log_dir),
                max_epochs=args.max_epochs,
                callbacks=[checkpoint_callback, MyTQDMProgressBar()],
            )
            trainer.fit(model, data_module)
            trainer.test(model, datamodule=data_module)
            latest_logfile = trainer.logger.log_dir
            print(f"Best score: {checkpoint_callback.best_model_score:.4f}")

            new_logfile = os.path.join(trainer.logger.log_dir[:-9], f"{checkpoint_callback.best_model_score:.4f}")
            if os.path.exists(new_logfile):
                new_logfile += f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.rename(latest_logfile, new_logfile)
            print(f"Finished run with frame_interval={frame_interval}, wd={wd}, dropout={dropout}")

    else:
        shutil.rmtree(args.cache_path, ignore_errors=True)
        os.makedirs(args.cache_path, exist_ok=True)
        seed_torch(args.seed)

        data_module = VideoRegressionDataModule(data_dir=args.data_dir,
                                                label_file=args.label_file,
                                                num_frames=args.num_frames,
                                                train_data=args.train_data,
                                                val_data=args.val_data,
                                                test_data=args.test_data,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                frame_interval=args.frame_interval,
                                                cache_path=args.cache_path,
                                                type=args.type)
        model = VideoRegressionModel(args)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_mae',
            mode='min',
            dirpath=f'best_models/',
            filename='{epoch}-{val_mae:.2f}-{val_rmse:.2f}'
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.deviceid,
            logger=CSVLogger(save_dir=f'logs/'),
            max_epochs=args.max_epochs,
            callbacks=[checkpoint_callback, MyTQDMProgressBar()],
        )
        trainer.fit(model, data_module)
        trainer.test(model, datamodule=data_module)
        latest_logfile = trainer.logger.log_dir
        print(f"Best score: {checkpoint_callback.best_model_score:.4f}")

        new_logfile = os.path.join(trainer.logger.log_dir[:-9], f"{checkpoint_callback.best_model_score:.4f}")
        if os.path.exists(new_logfile):
            new_logfile += f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.rename(latest_logfile, new_logfile)