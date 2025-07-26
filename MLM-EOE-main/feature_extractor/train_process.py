import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import numpy as np

from methods.IResNet.model import VideoFeatureExtractor


class VideoRegressionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.feature_extractor = VideoFeatureExtractor(dropout=args.dropout)
        self.mse = nn.MSELoss()
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()

        self.validation_step_outputs_pred = []
        self.validation_step_outputs_true = []
        self.validation_step_outputs_id = []
        self.args = args
        self.save_hyperparameters(args)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                               min_lr=1e-7,
                                                               verbose=True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.feature_extractor(x)
        train_loss = F.mse_loss(y_hat, y.reshape(-1, 1).float())
        train_mae = self.mean_absolute_error(y_hat, y.reshape(-1, 1).float())
        train_rmse = torch.sqrt(self.mean_squared_error(y_hat, y.reshape(-1, 1).float()))
        self.log('train_loss', train_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_mae', train_mae.item(), on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_rmse', train_rmse.item(), on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch)

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch)

    def _shared_eval_step(self, batch):
        x, y, video_ids = batch
        y_hat = self.feature_extractor(x)
        loss = F.mse_loss(y_hat, y.reshape(-1, 1).float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True, batch_size=y.shape[0])
        self.validation_step_outputs_pred.append(y_hat)
        self.validation_step_outputs_true.append(y)
        self.validation_step_outputs_id.append(video_ids)
        return y_hat, y, video_ids

    def _calculate_metrics(self):
        video_id_batches = self.validation_step_outputs_id
        all_video_ids = []
        for i in range(len(video_id_batches)):
            all_video_ids.extend(list(video_id_batches[i]))

        y_hat = torch.cat(self.validation_step_outputs_pred, dim=0)[:, 0]
        y_true = torch.cat(self.validation_step_outputs_true, dim=0)

        video_id_array = np.array(all_video_ids)
        mean_predictions = []
        mean_labels = []

        for unique_id in list(set(all_video_ids)):
            indices = np.where(video_id_array == unique_id)
            mean_predictions.append(torch.tensor(sorted(y_hat[indices]))[9:-10].mean().item())
            mean_labels.append(y_true[indices].mean().item())

        mae = self.mean_absolute_error(torch.tensor(mean_predictions), torch.tensor(mean_labels))
        rmse = torch.sqrt(self.mean_squared_error(torch.tensor(mean_predictions), torch.tensor(mean_labels)))

        self.validation_step_outputs_pred.clear()
        self.validation_step_outputs_true.clear()
        self.validation_step_outputs_id.clear()

        self.log('val_mae', mae.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_rmse', rmse.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        return {'val_mae': mae.item(), 'val_rmse': rmse.item()}

    def on_test_epoch_end(self):
        return self._calculate_metrics()

    def on_validation_epoch_end(self):
        return self._calculate_metrics()