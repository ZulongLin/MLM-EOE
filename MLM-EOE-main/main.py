import shutil
import sys
import os
import argparse
from itertools import product
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from data_loader.dataset_time_series import MILRegressionDataModule
from lightning_model import MILRegressionModel
import datetime

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    train_dataset = 'AVEC2014'
    test_dataset = 'AVEC2014'

    parser = argparse.ArgumentParser(description='Model Training Parameter Settings')

    parser.add_argument('--data_dir', default='', type=str,
                        help='Data storage directory')
    parser.add_argument('--label_file', default='', type=str,
                        help='Label file path')
    parser.add_argument('--train_data', default=[f'{train_dataset}-train'], nargs='+', help='Training dataset(s)')
    parser.add_argument('--val_data', default=[f'{test_dataset}-dev'], nargs='+', help='Validation dataset(s)')
    parser.add_argument('--test_data', default=[f'{test_dataset}-test'], nargs='+', help='Test dataset(s)')

    parser.add_argument('--model_name', default='Expert', help='Model name')
    parser.add_argument('--num_frames', default=1, type=int, help='Number of frames')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--max_epochs', default=50, type=int, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--frame_interval', default=1, type=int, help='Frame interval')
    parser.add_argument('--join', default=1, type=int, help='Join for two tasks of AVEC2014')

    parser.add_argument('--seed', type=float, default=1, help='Random seed')
    parser.add_argument('--devices', type=int, default=[0], nargs='+', help='GPU devices to use')
    parser.add_argument('--deep', default=True, action='store_true', help='Whether to use a deep model')
    parser.add_argument('--train_deep_type', default='iresnet50_base_1_new', type=str, help='Audio feature type')
    parser.add_argument('--test_deep_type', default='iresnet50_base_1_new', type=str, help='Audio feature type')

    parser.add_argument('--audio', default=False, type=bool, help='Whether to use audio features')
    parser.add_argument('--audio_type', default='audio_pann_64_3', type=str, help='Audio feature type')

    parser.add_argument('--rppg', default=False, type=bool, help='Whether to use rPPG features')
    parser.add_argument('--rppg_type', default='HRV_1000_normal', type=str, help='rPPG feature type')
    parser.add_argument('--emotion', default=False, type=bool, help='Whether to use emotion features')
    parser.add_argument('--emotion_type', default='Face_valence_arousal', type=str, help='Emotion feature type')

    parser.add_argument('--au', default=False, action='store_true', help='Whether to use AU features')
    parser.add_argument('--openface_type', default='openface_au', type=str, help='AU feature type')

    parser.add_argument('--use_inter', default=True, action='store_true', help='Input sequence length')
    parser.add_argument('--use_intra', default=True, action='store_true', help='Input sequence length')
    parser.add_argument('--use_patch_list', default=True, action='store_true', help='Input sequence length')

    parser.add_argument('--seq_len', type=int, default=2048, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction sequence length')
    parser.add_argument('--individual', action='store_true', default=True,
                        help='DLinear: individual linear layer for each channel')
    parser.add_argument('--d_model', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--num_nodes', type=int, default=21)
    parser.add_argument('--k', type=int, default=3, help='Select Top K patch sizes for each layer')
    parser.add_argument('--num_experts_list', type=list, default=[1, 2])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[32, 64, 128, 256])
    parser.add_argument('--revin', type=int, default=0, help='Whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.2, help='Dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time feature encoding, options: [timeF, fixed, learned]')
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--true_len', type=int, default=8192, help='Input sequence length after dividing by interval')
    parser.add_argument('--layer_nums', type=int, default=2)
    parser.add_argument('--time_index', type=int, default=1)
    parser.add_argument('--current_epoch', type=int, default=1)
    parser.add_argument('--is_training', type=bool, default=True)
    parser.add_argument('--distribute_by_score', type=bool, default=True, help='Whether to partition data by emotion')
    parser.add_argument('--USE_t_SNE', type=bool, default=False)

    args = parser.parse_args()
    args.true_len = int(args.seq_len / args.frame_interval)
    seed_everything(args.seed)

    data_module = MILRegressionDataModule(args)

    model = MILRegressionModel( args=args)

    early_stopping_callback = EarlyStopping(monitor='val_loss_epoch', mode='min', patience=15)
    args.layer_nums = args.num_experts_list.__len__()
    print(args)


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


    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_mae',
        mode='min',
        dirpath=f'best_model/{train_dataset}/',
        filename='{epoch}-{avg_val_mae:.4f}-{avg_val_rmse:.4f}'
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy='ddp_find_unused_parameters_true',
        logger=CSVLogger(
            save_dir=f'logs/{train_dataset}/'
        ),
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, MyTQDMProgressBar()],
        log_every_n_steps=15,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

