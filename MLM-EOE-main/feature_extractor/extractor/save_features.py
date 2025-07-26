import glob
from functools import lru_cache

import pandas as pd
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image
import argparse
import os
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import iresnet

torch.cuda.set_device(0)


@lru_cache(maxsize=None)
def load_image(path):
    return Image.open(path)


def load_frames(image_paths):
    frames = []
    for image in image_paths:
        frames.append(load_image(image))
    return frames


def load_state_dict_by_order_and_shape(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    checkpoint_state_dict = checkpoint['state_dict']

    model_state_dict = model.state_dict()

    model_params = list(model_state_dict.values())
    checkpoint_params = list(checkpoint_state_dict.values())

    model_param_names = list(model_state_dict.keys())
    checkpoint_param_names = list(checkpoint_state_dict.keys())

    num_model_params = len(model_params)
    num_checkpoint_params = len(checkpoint_params)

    matched_count = 0
    unmatched_model_params = []
    unmatched_checkpoint_params = []

    model_idx = 0
    checkpoint_idx = 0

    while model_idx < num_model_params and checkpoint_idx < num_checkpoint_params:
        model_param = model_params[model_idx]
        checkpoint_param = checkpoint_params[checkpoint_idx]

        if model_param.shape == checkpoint_param.shape:
            model_state_dict[model_param_names[model_idx]].copy_(checkpoint_param)
            matched_count += 1
            model_idx += 1
            checkpoint_idx += 1
        else:
            found_match = False
            for j in range(checkpoint_idx, num_checkpoint_params):
                if model_param.shape == checkpoint_params[j].shape:
                    model_state_dict[model_param_names[model_idx]].copy_(checkpoint_params[j])
                    matched_count += 1
                    model_idx += 1
                    checkpoint_idx = j + 1
                    found_match = True
                    break
            if not found_match:
                unmatched_model_params.append(model_param_names[model_idx])
                model_idx += 1

    while model_idx < num_model_params:
        unmatched_model_params.append(model_param_names[model_idx])
        model_idx += 1

    while checkpoint_idx < num_checkpoint_params:
        unmatched_checkpoint_params.append(checkpoint_param_names[checkpoint_idx])
        checkpoint_idx += 1

    print(f"Matched {matched_count} parameters.")
    if unmatched_model_params:
        print("Unmatched model parameters:")
        for name in unmatched_model_params:
            print(f"  - {name}")
    if unmatched_checkpoint_params:
        print("Unmatched checkpoint parameters:")
        for name in unmatched_checkpoint_params:
            print(f"- {name}")


class CustomDataset(Dataset):
    def __init__(self, sub_lists):
        self.sub_lists = sub_lists
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.sub_lists)

    def __getitem__(self, idx):
        image_paths = self.sub_lists[idx]
        image = load_image(image_paths)
        if self.transform:
            image = self.transform(image)
        return image


def map_weights(check_state_dict, model_state_dict):
    for i in range(len(list(model_state_dict.keys()))):
        model_state_dict[list(model_state_dict.keys())[i]] = check_state_dict[list(check_state_dict.keys())[i]]
    return model_state_dict


class FeatureExtractor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = iresnet.iresnet50(pretrained=False)
        self.model.features = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(self.model.fc.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        checkpoint_path = ''
        load_state_dict_by_order_and_shape(self.model, checkpoint_path)

    def forward(self, x):
        features = self.model(x)
        return features

    def training_step(self, batch, batch_idx):
        x, y = batch
        features = self(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction Settings')
    parser.add_argument('--data_dir', default='', type=str,
                        help='data_dir')
    parser.add_argument('--num_frames', default=1, type=int, help='num_frames')
    parser.add_argument('--dataname', default=['AVEC2014'], nargs='+', help='traindata')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
    parser.add_argument('--save_dir', default='', type=str,
                        help='save_dir', )
    parser.add_argument('--type', default='AVEC2013', type=str, help='type', )
    parser.add_argument('--save_file_name', default='iresnet50_base_1_pretrained', type=str, help='type', )
    args = parser.parse_args()
    # Enter the local tag file path
    labeldata = pd.read_csv('')
    model = FeatureExtractor().cuda()
    model.eval()

    for dataname in args.dataname:
        basepath = os.path.join(args.data_dir, dataname)
        all_filenames = glob.glob(os.path.join(basepath, '*_aligned'))
        filenames = []
        #Enter the local tag file path
        labeldata = pd.read_csv('')

        for i in trange(len(all_filenames)):
            filename = all_filenames[i]
            file_list = sorted(glob.glob(os.path.join(filename, '*.jpg')))

            savedir = os.path.join(args.save_dir, dataname + "_data", args.save_file_name)
            os.makedirs(savedir, exist_ok=True)
            savename = os.path.join(savedir, filename.split('/')[-1].replace('_aligned', '') + '.csv')

            dataset = CustomDataset(file_list)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True)

            output_list = []
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                output = model(data.cuda())
                output_list.append(output.data.cpu().numpy())

            combined_output = np.concatenate(output_list, axis=0)
            pd.DataFrame(combined_output).to_csv(savename, index=False)