import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from models.expert.models.Expert import Model as Expert

class MILRegressionModel(pl.LightningModule):
    def __init__(self, args):
        self.args = args
        super().__init__()

        ModelClass = globals()[args.model_name]
        self.model = ModelClass(configs=args)

        self.mse = nn.MSELoss()
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.validation_step_val_loss = []
        self.validation_step_val_pre = []
        self.validation_step_val_label = []

        self.epoch_start_time = None
        self.max_memory_allocated = 0
        self.epoch_times = []
        self.logged_metrics = []

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        time_index, x, y = batch
        self.args.time_index = time_index
        self.args.current_epoch = self.current_epoch
        y_hat = self(x.float())
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.track_memory()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch",
            },
        }

    def validation_step(self, batch, batch_idx):
        self.args.is_training = self.training
        self.evaluation_step(batch)
        self.track_memory()

    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch)
        self.track_memory()

    def on_validation_epoch_end(self):
        self.on_evaluation_epoch_end(stage='val')

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(stage='test')

    def evaluation_step(self, batch):
        time_index, x, y = batch
        self.args.time_index = time_index
        y_hat = self(x.float())

        loss = F.mse_loss(y_hat, y.view(-1, 1))

        [self.validation_step_val_pre.append(y_hat[i].item()) for i in range(y_hat.shape[0])]
        [self.validation_step_val_label.append(y[i].item()) for i in range(y.shape[0])]
        pd.DataFrame([self.validation_step_val_pre, self.validation_step_val_label]).T.to_csv(
            os.path.join(self.logger.log_dir, 'pre.csv'), index=False)
        self.validation_step_val_loss.append(loss)

    def on_evaluation_epoch_end(self, stage='val'):
        avg_val_loss = torch.stack([x for x in self.validation_step_val_loss]).mean()

        avg_val_mae = self.mean_absolute_error(torch.tensor(self.validation_step_val_pre),
                                               torch.tensor(self.validation_step_val_label))
        avg_val_rmse = torch.sqrt(self.mean_squared_error(torch.tensor(self.validation_step_val_pre),
                                                          torch.tensor(self.validation_step_val_label)))

        self.log_dict(
            {f'{stage}_loss_epoch': avg_val_loss, f'avg_{stage}_mae': avg_val_mae, f'avg_{stage}_rmse': avg_val_rmse},
            on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.validation_step_val_pre.clear()
        self.validation_step_val_label.clear()
        self.validation_step_val_loss.clear()

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        self.max_memory_allocated = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self):
        self.log_epoch_metrics()
        self.save_logged_metrics()

    def on_validation_epoch_start(self):
        if self.epoch_start_time is None:
            self.epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_test_epoch_start(self):
        if self.epoch_start_time is None:
            self.epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def track_memory(self):
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.max_memory_allocated = max(self.max_memory_allocated, max_memory)

    def log_epoch_metrics(self):
        metrics = {}
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_duration)
            metrics['epoch_duration'] = epoch_duration
            self.log('epoch_duration', epoch_duration, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if torch.cuda.is_available():
            metrics['max_memory_allocated'] = self.max_memory_allocated
            self.log('max_memory_allocated', self.max_memory_allocated, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)

        metrics['epoch'] = self.current_epoch
        self.logged_metrics.append(metrics)

        self.epoch_start_time = None

    def save_logged_metrics(self):
        if self.logged_metrics:
            df = pd.DataFrame(self.logged_metrics)
            filepath = os.path.join(self.logger.log_dir, 'epoch_metrics.csv')
            df.to_csv(filepath, index=False)