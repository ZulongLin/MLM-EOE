import os
import pickle
import random
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from tqdm import trange

import lightning as pl
import glob
import matplotlib.pyplot as plt


def get_info_by_index(labeldata, data_split):
    mask = (labeldata['dataname'] == data_split[0].split('-')[0]) & (labeldata['group'] == data_split[0].split('-')[1])
    result = labeldata[mask]

    for i in trange(1, len(data_split)):
        mask = (labeldata['dataname'] == data_split[i].split('-')[0]) & (
                    labeldata['group'] == data_split[i].split('-')[1])
        result = pd.concat([result, labeldata[mask]])
    return result


@lru_cache(maxsize=None)
def load_image(path):
    return Image.open(path)


def load_frames(image_paths):
    frames = []
    for image in image_paths:
        frames.append(load_image(image))
    return frames


class VideoDataset(Dataset):
    def __init__(self, data_dir, labeldata, data, num_frames=16, type="image", stage='test', frame_interval=2,
                 transform=None, cache_path=None):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.labeldata = labeldata
        self.transforms = transform
        self.result = []

        cachefile = os.path.join(cache_path, stage + '.pkl')

        result = get_info_by_index(labeldata, data).values
        for i in trange(result.shape[0]):
            filename = result[i][3][:-4]
            file_list = sorted(glob.glob(os.path.join(data_dir, type, filename + '_aligned', '*')))
            file_list = file_list[::frame_interval]
            sub_lists = [file_list[i:i + num_frames] for i in
                         range(0, len(file_list) - len(file_list) % num_frames, num_frames)]
            self.result = self.result + sub_lists

        print('Data loading complete')
        self.cache_path = os.path.join(cache_path, stage)
        if self.cache_path and not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        image_paths = self.result[idx]
        frames = load_frames(image_paths)
        frames = torch.stack([torch.from_numpy(np.array(frame)) for frame in frames])

        if self.transforms:
            frames = self.transforms(frames.permute(0, 3, 1, 2).float())

        score = self.labeldata[self.labeldata['filename'] == image_paths[0].split('/')[-2][:-8] + '.mp4'].values[0][2]
        return frames.permute(1, 0, 2, 3), torch.tensor(score).float(), image_paths[0].split('/')[-2]


class VideoRegressionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, label_file, train_data, val_data, test_data, num_frames=16, batch_size=32,
                 num_workers=4, frame_interval=2, type='image', cache_path=None):
        super().__init__()
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.labeldata = pd.read_csv(label_file)

        self.train_dataset = VideoDataset(data_dir, self.labeldata, train_data, num_frames=num_frames, type=type,
                                          stage='train', frame_interval=frame_interval, transform=train_transforms,
                                          cache_path=cache_path)
        self.val_dataset = VideoDataset(data_dir, self.labeldata, val_data, num_frames=num_frames, type=type,
                                        stage='dev', frame_interval=frame_interval, transform=val_transform,
                                        cache_path=cache_path)
        self.test_dataset = VideoDataset(data_dir, self.labeldata, test_data, num_frames=num_frames, type=type,
                                        stage='test', frame_interval=frame_interval, transform=val_transform,
                                        cache_path=cache_path)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size,
                                num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers)
        return test_loader