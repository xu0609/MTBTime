# 文件 2：dataset.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from timefeatures import time_features
import pandas as pd


def load_raw_data(dataset_config):
    """
Load the original dataset and perform preprocessing

       """

    if 'PEMS' in dataset_config.dataset_name:

        raw_data = np.load(dataset_config.data_filename)['data']

        # Divide the dataset in a 6:2:2 ratio
        train_data_seq = raw_data[:int(0.6 * raw_data.shape[0])]#
        val_data_seq = raw_data[int(0.6 * raw_data.shape[0]):int(0.8 * raw_data.shape[0])]#
        test_data_seq = raw_data[int(0.8 * raw_data.shape[0]):]

        train_mean = np.mean(train_data_seq, axis=(0, 1))#
        train_std = np.std(train_data_seq, axis=(0, 1)) #

        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        return train_mean, train_std, train_data_seq, test_data_seq



    elif dataset_config.dataset_name == 'ETTm1' or dataset_config.dataset_name == 'Weather':

        raw_data = pd.read_csv(dataset_config.data_filename)


        raw_data_feats = raw_data.values[:, 1:]#
        raw_data_stamps = raw_data.values[:, 0]#
        raw_data_stamps = pd.to_datetime(raw_data_stamps)


        train_data_seq = raw_data_feats[:int(0.6 * raw_data_feats.shape[0])]
        val_data_seq = raw_data_feats[int(0.6 * raw_data_feats.shape[0]):int(0.8 * raw_data_feats.shape[0])]
        test_data_seq = raw_data_feats[int(0.8 * raw_data_feats.shape[0]):]



        train_data_stamps = raw_data_stamps[:int(0.6 * raw_data_stamps.shape[0])]
        val_data_stamps = raw_data_stamps[int(0.6 * raw_data_stamps.shape[0]):int(0.8 * raw_data_stamps.shape[0])]
        test_data_stamps = raw_data_stamps[int(0.8 * raw_data_stamps.shape[0]):]


        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]


        return train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps

    else:
        raise ValueError('Dataset not supported')


class TimeDataset(Dataset):
    def __init__(self, raw_data, mean, std, device, num_for_hist=12, num_for_futr=12, timestamps=None):
        self.device = device
        self.data = raw_data
        self.use_timestamp = timestamps is not None


        if self.use_timestamp:
            self.timestamps = time_features(timestamps)
            self.timestamps = self.timestamps.transpose(1, 0)
            self.timestamps = torch.from_numpy(self.timestamps).float().to(self.device)
        else:
            self.timestamps = None

        # Data dimension processing: Ensure that the data is in 3D format (number of nodes, features, time steps)
        if len(self.data.shape) == 2:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1)
        self.data = np.transpose(self.data, (1, 2, 0)).astype(np.float32)
        self.data = torch.from_numpy(self.data).float().to(self.device)

        self.init_poison_data()

        self.std = float(std)
        self.mean = float(mean)

        #
        self.num_for_hist = num_for_hist
        self.num_for_futr = num_for_futr

    def __len__(self):
        return self.data.shape[-1] - self.num_for_hist - self.num_for_futr + 1

    def __getitem__(self, idx):
        data = self.poisoned_data[:, 0:1, idx:idx + self.num_for_hist]
        data = self.normalize(data)

        poisoned_target = self.poisoned_data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]

        clean_target = self.data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
        if not self.use_timestamp:
            return data, poisoned_target, clean_target, idx
        else:
            input_stamps = self.timestamps[idx:idx + self.num_for_hist]
            target_stamps = self.timestamps[idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
            return data, poisoned_target, clean_target, input_stamps, target_stamps, idx

    def init_poison_data(self):

        self.poisoned_data = torch.clone(self.data).detach().to(self.device)

    def normalize(self, data):

        return (data - self.mean) / self.std

    def denormalize(self, data):

        return data * self.std + self.mean


class AttackEvaluateSet(TimeDataset):

    def __init__(self, attacker, raw_data, mean, std, device, num_for_hist=12, num_for_futr=12, timestamps=None):

        #
        super(AttackEvaluateSet, self).__init__(raw_data, mean, std, device, num_for_hist, num_for_futr, timestamps)

        #
        self.attacker = attacker

    def collate_fn(self, data):
        """A data batch processing function used to generate evaluation data after an attack
        (supporting different target patterns for each variable)"""

        if self.use_timestamp:
            features, target, clean_target, input_stamps, target_stamps, idx = zip(*data)
            input_stamps = torch.stack(input_stamps, dim=0)
            target_stamps = torch.stack(target_stamps, dim=0)
        else:
            features, target, clean_target, idx = zip(*data)

        features = torch.stack(features, dim=0)  # (B, N, C=1, L_hist)
        clean_target = torch.stack(clean_target, dim=0)  # (B, N, L_futr)


        features = self.denormalize(features)
        data_bef = features[:, self.attacker.atk_vars, 0,
                   -self.attacker.trigger_len - self.attacker.bef_tgr_len: -self.attacker.trigger_len]
        triggers = self.attacker.predict_trigger(data_bef)[0]
        triggers = triggers.reshape(-1, self.attacker.atk_vars.shape[0], 1, self.attacker.trigger_len)
        features[:, self.attacker.atk_vars, :, -self.attacker.trigger_len:] = triggers


        target = clean_target.clone().detach().to(self.device)
        last_val = features[:, self.attacker.atk_vars, :, -self.attacker.trigger_len - 1]  # (B, n_atk, 1)

        ptn = self.attacker.target_pattern
        if ptn.dim() == 1:
            # Single mode: (L,) -> (1,1,L)
            ptn = ptn.view(1, 1, -1)
        elif ptn.dim() == 2:
            # Per variable mode: (n_atk, L) -> (1, n_atk, L)
            ptn = ptn.unsqueeze(0)
        elif ptn.dim() == 3:
            #  (n_atk,1,L) -> (1, n_atk, L)
            if ptn.shape[1] == 1:
                ptn = ptn.permute(1, 0, 2)
        else:
            raise ValueError("Unexpected target_pattern shape")

        target[:, self.attacker.atk_vars, : self.attacker.pattern_len] = ptn + last_val  # -> (B, n_atk, L_ptn)

        #
        features = self.normalize(features)
        if not self.use_timestamp:
            return features, target, clean_target, idx
        else:
            return features, target, clean_target, input_stamps, target_stamps, idx
