import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import trange, tqdm
import lightning as pl
import glob

np.set_printoptions(suppress=True)

def process_scientific_notation(matrix):
    result = np.empty_like(matrix)
    for index, value in np.ndenumerate(matrix):
        if isinstance(value, str) and '-' == value.lower():
            print(value)
            result[index] = 0
        else:
            result[index] = value
    return result

class Normalizer(object):
    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)
        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')
        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)
        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

def get_info_by_split(label_df, data_split):
    mask = (label_df['dataname'] == data_split[0].split('-')[0]) & (label_df['group'] == data_split[0].split('-')[1])
    result = label_df[mask]
    for i in range(1, len(data_split)):
        mask = (label_df['dataname'] == data_split[i].split('-')[0]) & (label_df['group'] == data_split[i].split('-')[1])
        result = pd.concat([result, label_df[mask]])
    return result

def subtract_median(arr):
    median = np.median(arr, axis=0)
    return arr - median

def adjust_matrix_columns(matrix, dims):
    original_cols = matrix.shape[1]
    if original_cols < dims:
        pad_cols = dims - original_cols
        matrix = np.pad(matrix, ((0, 0), (0, pad_cols)), mode='constant')
    elif original_cols > dims:
        matrix = matrix[:, :dims]
    return np.array(matrix, dtype=np.float32)

def adjust_matrix_columns_v2(matrix, dims):
    original_cols = matrix.shape[1]
    if original_cols < dims:
        pad_cols = dims - original_cols
        fill_data = matrix[:, :original_cols]
        while fill_data.shape[1] < pad_cols:
            fill_data = np.hstack([fill_data, matrix[:, :original_cols]])
        matrix = np.hstack([matrix, fill_data[:, :pad_cols]])
    elif original_cols > dims:
        matrix = matrix[:, :dims]
    return matrix

def read_multiple_csvs(csv_files):
    data = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        data = pd.concat([data, df], axis=0, ignore_index=True)
    return data

class MILDataset(Dataset):
    def __init__(self, args, label_df, data_split, stage=None):
        self.data_dir = args.data_dir
        self.stage = stage
        self.args = args
        self.label_df = label_df
        self.frame_interval = args.frame_interval
        self.data_list = []
        self.labels = []
        self.names = []
        self.norm = Normalizer(mean=0, std=1)
        self.input_dims = 0
        result = get_info_by_split(label_df, data_split).values

        for i in trange(result.shape[0]):
            self.labels.append(label_df[label_df['filename'] == result[i][3]]['lablescore'].values[0])

            if args.join == 1:
                filename = result[i][3][:5]
            else:
                filename = result[i][3][:-5]

            if filename in self.names:
                continue
            self.names.append(filename)

            concatenated_data = pd.DataFrame()
            min_len = float('inf')

            if args.deep:
                csv_files = sorted(
                    glob.glob(os.path.join(args.data_dir, result[i][0] + "_data", args.train_deep_type, filename + '*.csv')))
                if self.stage != 'train':
                    csv_files = sorted(
                        glob.glob(os.path.join(args.data_dir, result[i][0] + "_data", args.test_deep_type,
                                               filename + '*.csv')))
                deep_data = read_multiple_csvs(csv_files)
                if 'AVEC2017' in args.train_data[0]:
                    deep_columns = [col for col in deep_data.columns if
                                    ("x" in col) or ('y' in col) or ('z' in col)]
                    deep_data = deep_data[deep_columns]
                min_len = min(min_len, deep_data.shape[0])
                concatenated_data = pd.concat([concatenated_data, deep_data[:min_len].fillna(0)], axis=1)
                args.video_dims = deep_data.shape[1]

            if args.emotion:
                csv_files = sorted(
                    glob.glob(
                        os.path.join(args.data_dir, result[i][0] + "_data", args.emotion_type, filename + '*.csv')))
                emotion_data = read_multiple_csvs(csv_files).iloc[:, 0:2]
                min_len = min(min_len, emotion_data.shape[0])
                concatenated_data = pd.concat([concatenated_data, emotion_data[:min_len]], axis=1)

            if args.audio:
                csv_files = sorted(
                    glob.glob(
                        os.path.join(args.data_dir, result[i][0] + "_data", args.audio_type, filename + '*.csv')))
                audio_data = read_multiple_csvs(csv_files)
                if 'AVEC2019' in self.args.train_data[0]:
                    audio_data = audio_data.iloc[:, 1:]
                min_len = min(min_len, audio_data.shape[0])
                concatenated_data = pd.concat([concatenated_data, audio_data[:min_len]], axis=1)
                args.audio_dims = audio_data.shape[1]

            if args.au:
                csv_files = sorted(
                    glob.glob(
                        os.path.join(args.data_dir, result[i][0] + "_data", args.openface_type,
                                     filename + '*.csv')))
                openface_data = read_multiple_csvs(csv_files)
                au_columns = [col for col in openface_data.columns if
                              ('AU' in col and "_r" in col) or ('gaze' in col and 'angle' not in col) or (
                                          'pose' in col)]
                openface_data = openface_data[au_columns]
                if not concatenated_data.empty:
                    last_col = concatenated_data.columns[-1]
                else:
                    last_col = None
                min_len = min(min_len, openface_data.shape[0])
                concatenated_data = pd.concat([concatenated_data, openface_data[:min_len]], axis=1)
                args.au_dims = openface_data.shape[1]
                if last_col is not None and last_col == 'score':
                    cols = [col for col in concatenated_data.columns if col != last_col] + [last_col]
                    concatenated_data = concatenated_data[cols]

            if args.rppg:
                csv_files = sorted(
                    glob.glob(os.path.join(args.data_dir, result[i][0] + "_data", args.rppg_type, filename + '*.csv')))
                if len(csv_files) == 0:
                    continue
                rppg_data = read_multiple_csvs(csv_files).iloc[1:, 1:]
                min_len = min(min_len, rppg_data.shape[0])
                concatenated_data = pd.concat(
                    [concatenated_data, rppg_data.iloc[:min_len].fillna(0)], axis=1)
                args.rppg_dims = rppg_data.shape[1]

            concatenated_data.fillna(0, inplace=True)
            self.args.num_modals = sum([args.deep, args.audio, args.rppg, args.au])
            self.data_list.append(concatenated_data)
            self.args.input_dims = self.data_list[0].values.shape[1]
            args.num_nodes = self.data_list[0].values.shape[1] - 1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx].values.T
        label = self.labels[idx]
        data = adjust_matrix_columns(data, self.args.seq_len).T
        data = data[::self.args.frame_interval]
        data = torch.tensor(data).float()
        return idx, data, torch.tensor(label).float()

class MILRegressionDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.frame_interval = args.frame_interval
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.input_dims = 0
        self.label_df = pd.read_csv(args.label_file)
        self.train_dataset = MILDataset(args, self.label_df, args.train_data, stage='train')
        self.val_dataset = MILDataset(args, self.label_df, args.val_data, stage='dev')
        self.test_dataset = MILDataset(args, self.label_df, args.test_data, stage='test')
        self.input_dims = self.val_dataset.input_dims
        self.save_hyperparameters(args)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True, drop_last=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True, drop_last=True)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers, drop_last=True)
        return test_loader