import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm
import numpy as np
import pandas as pd
from dataset import TimeDataset, AttackEvaluateSet
from attack import Attacker, fft_compress
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forecast_models import TimesNet, Autoformer, FEDformer
from torch.utils.data import DataLoader  # 添加此行
from forecast_models.TSMixer import Model as TSMixer
MODEL_MAP = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'FEDformer': FEDformer,
    'TSMixer' :  TSMixer

}


class Trainer:
    def __init__(self, config, atk_vars, target_pattern, train_mean, train_std,
                 train_data, test_data, train_data_stamps, test_data_stamps, device):
        self.config = config
        self.mean = train_mean
        self.std = train_std
        self.test_data = test_data
        self.net = MODEL_MAP[config.surrogate_name](config.Surrogate).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.device = device

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.warmup = config.warmup

        self.train_data_stamps = train_data_stamps
        self.test_data_stamps = test_data_stamps

        train_set = TimeDataset(train_data, train_mean, train_std, device, num_for_hist=12, num_for_futr=12,
                                timestamps=train_data_stamps)
        channel_features = fft_compress(train_data, 200)
        self.attacker = Attacker(train_set, channel_features, atk_vars, config, target_pattern, device)
        self.use_timestamps = config.Dataset.use_timestamps

        self.prepare_data()

        # Initialize metrics storage
        self.metrics = {
            'clean_mae': [],
            'clean_rmse': [],
            'attacked_mae': [],
            'attacked_rmse': []
        }

    def load_attacker(self, attacker_state):
        self.attacker.load_state_dict(attacker_state)

    def save_attacker(self):
        attacker_state = self.attacker.state_dict()
        return attacker_state

    def prepare_data(self):
        self.train_set = self.attacker.dataset
        self.cln_test_set = TimeDataset(self.test_data, self.mean, self.std, self.device, num_for_hist=12,
                                        num_for_futr=12, timestamps=self.test_data_stamps)
        self.atk_test_set = AttackEvaluateSet(self.attacker, self.test_data, self.mean, self.std, self.device,
                                              num_for_hist=12, num_for_futr=12, timestamps=self.test_data_stamps)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.cln_test_loader = DataLoader(self.cln_test_set, batch_size=self.batch_size, shuffle=False)
        self.atk_test_loader = DataLoader(self.atk_test_set, batch_size=self.batch_size, shuffle=False,
                                          collate_fn=self.atk_test_set.collate_fn)

        # Initialize metrics storage
        self.metrics = {
            'clean_mae': [],
            'clean_rmse': [],
            'attacked_mae': [],
            'attacked_rmse': []
        }

    def train(self):


        self.attacker.train()

        poison_metrics = []

        for epoch in range(self.num_epochs):

            self.net.train()  # ensure dropout layers are in train mode

            if epoch > self.warmup:

                if not hasattr(self.attacker, 'atk_ts'):

                    # select the attacked timestamps
                    self.attacker.select_atk_timestamp(poison_metrics)

                # attacker poison the training data
                self.attacker.sparse_inject()

            poison_metrics = []

            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

            pbar = tqdm.tqdm(self.train_loader, desc=f'Training data {epoch}/{self.num_epochs}')

            for batch_index, batch_data in enumerate(pbar):
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                self.optimizer.zero_grad()

                x_des = torch.zeros_like(labels)

                outputs = self.net(encoder_inputs, x_mark, x_des, None)

                outputs = self.train_set.denormalize(outputs)

                loss_per_sample = F.smooth_l1_loss(outputs, labels, reduction='none')

                loss_per_sample = loss_per_sample.mean(dim=(1, 2))

                mae_per_sample = torch.abs(outputs - labels).mean(dim=(1, 2))

                poison_metrics.append(torch.stack([mae_per_sample.cpu().detach(), idx.cpu().detach()], dim=1))

                loss = loss_per_sample.mean()

                loss.backward()

                self.optimizer.step()

            if epoch > self.warmup:
                self.attacker.update_trigger_generator(self.net, epoch, self.num_epochs,
                                                       use_timestamps=self.use_timestamps)

            self.validate(self.net, epoch, self.warmup)

    def validate(self, model, epoch, atk_eval_epoch=0):

        """
           Validation function - Evaluate the performance of the model on clean and attacked data

           Args:
               model: The model to be validated
               epoch: Current training round
               atk_eval_epoch: Start evaluating the round threshold for attack performance
           """
        model.eval()
        self.attacker.eval()
        cln_info = atk_info = ''
        with torch.no_grad():
            cln_preds = []
            atk_preds = []
            cln_targets = []
            atk_targets = []

            for batch_index, batch_data in enumerate(self.cln_test_loader):
                # calculate the clean performance
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                x_des = torch.zeros_like(labels)
                outputs = model(encoder_inputs, x_mark, x_des, None)
                outputs = self.cln_test_set.denormalize(outputs)
                cln_targets.append(labels.cpu().detach().numpy())
                cln_preds.append(outputs.cpu().detach().numpy())

            cln_preds = np.concatenate(cln_preds, axis=0)
            cln_targets = np.concatenate(cln_targets, axis=0)
            cln_mae = mean_absolute_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1))
            cln_rmse = mean_squared_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1)) ** 0.5

            cln_info = f' | clean MAE: {cln_mae}, clean RMSE: {cln_rmse}'

            if epoch > atk_eval_epoch:
                for batch_index, batch_data in enumerate(self.atk_test_loader):
                    # calculate the attacked performance
                    if not self.use_timestamps:
                        encoder_inputs, labels, clean_labels, idx = batch_data
                        x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                    else:
                        encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                    labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                    x_des = torch.zeros_like(labels)
                    outputs = model(encoder_inputs, x_mark, x_des, None)
                    outputs = self.atk_test_set.denormalize(outputs)

                    labels = labels[:, :self.attacker.pattern_len, self.attacker.atk_vars]
                    outputs = outputs[:, :self.attacker.pattern_len, self.attacker.atk_vars]
                    atk_targets.append(labels.cpu().detach().numpy())
                    atk_preds.append(outputs.cpu().detach().numpy())

                atk_preds = np.concatenate(atk_preds, axis=0)
                atk_targets = np.concatenate(atk_targets, axis=0)
                atk_mae = mean_absolute_error(atk_targets.reshape(-1, 1), atk_preds.reshape(-1, 1))
                atk_rmse = mean_squared_error(atk_targets.reshape(-1, 1), atk_preds.reshape(-1, 1)) ** 0.5

                atk_info = f' | attacked MAE: {atk_mae}, attacked RMSE: {atk_rmse}'

        info = 'Epoch: {}'.format(epoch) + cln_info + atk_info
        print(info)

    def compute_average_metrics(self):
        avg_clean_mae = np.mean(self.metrics['clean_mae'])
        avg_clean_rmse = np.mean(self.metrics['clean_rmse'])
        avg_attacked_mae = np.mean(self.metrics['attacked_mae'])
        avg_attacked_rmse = np.mean(self.metrics['attacked_rmse'])
        return {
            'avg_clean_mae': avg_clean_mae,
            'avg_clean_rmse': avg_clean_rmse,
            'avg_attacked_mae': avg_attacked_mae,
            'avg_attacked_rmse': avg_attacked_rmse
        }

    # trainer.py
    def test(self):

        self.attacker.eval()


        model = MODEL_MAP[self.config.model_name](self.config.Model).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)


        self.attacker.sparse_inject()

        self.train_set = self.attacker.dataset

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)


        for epoch in range(self.num_epochs):

            pbar = tqdm.tqdm(self.train_loader, desc=f'Training new forecasting model {epoch}/{self.num_epochs}')


            for batch_index, batch_data in enumerate(pbar):

                if not self.use_timestamps:

                    encoder_inputs, labels, clean_labels, idx = batch_data

                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:

                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data


                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)

                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)


                optimizer.zero_grad()


                x_des = torch.zeros_like(labels)

                outputs = model(encoder_inputs, x_mark, x_des, None)

                outputs = self.train_set.denormalize(outputs)


                loss = F.smooth_l1_loss(outputs, labels)

                loss.backward()

                optimizer.step()


            self.validate(model, epoch, 0)
