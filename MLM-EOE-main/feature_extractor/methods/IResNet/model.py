import os
import sys
import torch.nn as nn

from methods.IResNet.backbones import iresnet

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


class VideoFeatureExtractor(nn.Module):
    def __init__(self, dropout=0.7, pretrained=True):
        super().__init__()
        self.model = iresnet.iresnet50(pretrained=False)
        # load your checkpoint here

        additional_layers = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(self.model.fc.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.model.features = additional_layers
        print(self.model)

    def forward(self, x):
        x = x.squeeze(2)
        x = self.model(x)
        return x
