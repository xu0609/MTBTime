import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import optim
import numpy as np
import pandas as pd
from math import ceil
import tqdm
from trigger import TgrGCN
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from Cau import CausalDiscoveryModule
def fft_compress(raw_data_seq, n_components=200):
    """
    compress the time series data using fft to have global representation for each variable.
    """
    if len(raw_data_seq.shape) == 2:
        raw_data_seq = raw_data_seq[:, :, None]
    data_seq = raw_data_seq[:, :, 0:1]
    # data_seq: (l, n, c)
    l, n, c = data_seq.shape
    data_seq = data_seq.reshape(l, -1).transpose()
    # use fft to have the amplitude, phase, and frequency for each time series data
    fft_data = np.fft.fft(data_seq, axis=1)
    amplitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    frequency = np.fft.fftfreq(l)

    # choose the top n_components frequency components
    top_indices = np.argsort(amplitude, axis=1)[::-1][:, :n_components]
    amplitude_top = amplitude[np.arange(amplitude.shape[0])[:, None], top_indices]
    phase_top = phase[np.arange(phase.shape[0])[:, None], top_indices]
    frequency_top = frequency[top_indices]
    feature_top = np.concatenate([amplitude_top, phase_top, frequency_top], axis=1)
    return feature_top


class Attacker:
    def __init__(self, dataset, channel_features, atk_vars, config, target_pattern, device='cuda'):
        """
        the attacker class is used to inject triggers and target patterns into the dataset.
        the attacker class have the full access to the dataset and the trigger generator.
        """
        self.device = device
        self.dataset = dataset

        self.target_pattern = target_pattern
        self.atk_vars = atk_vars

        self.trigger_generator = TgrGCN(config, sim_feats=channel_features, atk_vars=atk_vars, device=device)
        self.trigger_len = config.trigger_len
        self.pattern_len = config.pattern_len
        self.bef_tgr_len = config.bef_tgr_len  # the length of the data before the trigger to generate the trigger

        # === causal prior (optional) ===
        self.causal_prior_A = None  # shape: [num_vars, num_vars]
        self.causal_prior_save = getattr(config, 'causal_prior_save', 'causal_prior.npy')


        self.fct_input_len = config.Dataset.len_input  # the length of the input for the forecast model
        self.fct_output_len = config.Dataset.num_for_predict  # the length of the output for the forecast model
        self.alpha_t = config.alpha_t
        self.alpha_s = config.alpha_s
        self.temporal_poison_num = ceil(self.alpha_t * len(self.dataset))

        self.trigger_generator = self.trigger_generator.to(device)
        self.attack_optim = optim.Adam(self.trigger_generator.parameters(), lr=config.attack_lr)
        self.atk_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.attack_optim, milestones=[20, 40], gamma=0.9)

        self.lam_norm = config.lam_norm


        self.causal_fusion = config.get('causal_fusion', 'weighted')  # 'weighted' 或 'gated'
        self.causal_w_neural = float(config.get('causal_w_neural', 0.8))  # weighted下神经分数权重
        self.causal_gate_bias = float(config.get('causal_gate_bias', 0.2))  # gated下最小门值
        self.causal_granger_maxlag = int(config.get('causal_granger_maxlag', 2))  # Granger最大滞后阶
        self.causal_window_size = int(config.get('causal_window_size', 12))  # 滑窗大小


        self.num_vars = self.dataset.data.shape[0]
        self.in_dim = self.num_vars * self.dataset.data.shape[1]  # n*c
        self.causal_module = CausalDiscoveryModule(
            num_vars=self.num_vars,
            embed_dim=32,
            in_dim=self.in_dim,  # 输入特征长度
            top_k=10
        ).to(self.device)


    def state_dict(self):
        attacker_state = {
            'target_pattern': self.target_pattern.cpu().detach().numpy(),
            'trigger_generator': self.trigger_generator.state_dict(),
            'trigger_len': self.trigger_len,
            'pattern_len': self.pattern_len,
            'bef_tgr_len': self.bef_tgr_len,
            'fct_input_len': self.fct_input_len,
            'fct_output_len': self.fct_output_len,
            'alpha_t': self.alpha_t,
            'alpha_s': self.alpha_s,
            'temporal_poison_num': self.temporal_poison_num,
            'lam_norm': self.lam_norm,
            'attack_optim': self.attack_optim.state_dict(),
            'atk_scheduler': self.atk_scheduler.state_dict(),
            'atk_ts': self.atk_ts.cpu().detach().numpy() if hasattr(self, 'atk_ts') else None,
            'atk_vars': self.atk_vars.cpu().detach().numpy() if hasattr(self, 'atk_vars') else None,
        }
        return attacker_state

    def load_state_dict(self, attacker_state):
        # Load basic configuration
        self.trigger_len = attacker_state['trigger_len']
        self.pattern_len = attacker_state['pattern_len']
        self.bef_tgr_len = attacker_state['bef_tgr_len']
        self.fct_input_len = attacker_state['fct_input_len']
        self.fct_output_len = attacker_state['fct_output_len']
        self.alpha_t = attacker_state['alpha_t']
        self.alpha_s = attacker_state['alpha_s']
        self.temporal_poison_num = attacker_state['temporal_poison_num']
        self.lam_norm = attacker_state['lam_norm']

        # 1. Load model parameters first (compatible with missing keys)
        model_state = self.trigger_generator.state_dict()
        saved_model_state = attacker_state['trigger_generator']
        for key in model_state:
            if key in saved_model_state:
                model_state[key] = saved_model_state[key]
        self.trigger_generator.load_state_dict(model_state)

        # 2. Rebuilding optimizer (ensuring parameter set matches current model)
        self.attack_optim = optim.Adam(
            self.trigger_generator.parameters(),
            lr=attacker_state['attack_optim']['param_groups'][0]['lr']
        )

        # 3. Load optimizer state (only load matching parameters)
        optimizer_state = attacker_state['attack_optim']
        current_optim_state = self.attack_optim.state_dict()

        # Copy matching parameter states
        for param in current_optim_state['state']:
            if param in optimizer_state['state']:
                current_optim_state['state'][param] = optimizer_state['state'][param]
        self.attack_optim.load_state_dict(current_optim_state)

        # Load other states
        self.atk_scheduler.load_state_dict(attacker_state['atk_scheduler'])
        self.target_pattern = torch.from_numpy(attacker_state['target_pattern']).to(self.device)
        self.atk_ts = torch.from_numpy(attacker_state['atk_ts']).to(self.device) if attacker_state[
                                                                                        'atk_ts'] is not None else None
        self.atk_vars = torch.from_numpy(attacker_state['atk_vars']).to(self.device) if attacker_state[
                                                                                            'atk_vars'] is not None else None

    def eval(self):
        self.trigger_generator.eval()

    def train(self):
        self.trigger_generator.train()

    def set_atk_timestamp(self, atk_ts):
        """
        set the attack timestamp for the attacker.
        """
        self.atk_ts = atk_ts

    def set_atk_variables(self, atk_var):
        """
        set the attack variables for the attacker.
        """
        self.atk_vars = atk_var

    def set_atk(self, atk_ts, atk_var):
        self.set_atk_timestamp(atk_ts)
        self.set_atk_variables(atk_var)

    def dense_inject(self):
        """
        Inject the trigger and target pattern into all the variables at the attack timestamp.
        This function has been deprecated. please consider sparse_inject() instead.
        """
        assert hasattr(self, 'atk_ts'), 'Please set the attack timestamp first.'
        self.dataset.init_poison_data()

        n, c, T = self.dataset.data.shape
        for beg_idx in self.atk_ts.tolist():
            data_bef_tgr = self.dataset.data[..., beg_idx - self.trigger_generator.input_dim:beg_idx]
            data_bef_tgr = self.dataset.normalize(data_bef_tgr)
            data_bef_tgr = data_bef_tgr.view(-1, self.trigger_generator.input_dim)

            triggers = self.trigger_generator(data_bef_tgr)[0]
            triggers = self.dataset.denormalize(triggers).reshape(n, c, -1)

            self.dataset.poisoned_data[..., beg_idx:beg_idx + self.trigger_len] = triggers.detach()
            self.dataset.poisoned_data[..., beg_idx + self.trigger_len:beg_idx + self.trigger_len + self.pattern_len] = \
                self.target_pattern + self.dataset.poisoned_data[..., beg_idx - 1:beg_idx]

    def sparse_inject(self):
        """
        Inject the trigger and target pattern into all the variables at the attack timestamp.
        """
        assert hasattr(self, 'atk_vars'), 'Please set the attack variable first.'
        assert hasattr(self, 'atk_ts'), 'Please set the attack timestamp first.'
        self.dataset.init_poison_data()

        n, c, T = self.dataset.data.shape
        n = len(self.atk_vars)
        trigger_len = self.trigger_generator.output_dim
        pattern_len = self.target_pattern.shape[-1]

        for beg_idx in self.atk_ts.tolist():
            data_bef_tgr = self.dataset.data[self.atk_vars, 0:1, beg_idx - self.trigger_generator.input_dim:beg_idx]
            data_bef_tgr = self.dataset.normalize(data_bef_tgr)
            data_bef_tgr = data_bef_tgr.reshape(-1, self.trigger_generator.input_dim)

            triggers = self.trigger_generator(data_bef_tgr)[0]
            triggers = self.dataset.denormalize(triggers).reshape(n, 1, -1)

            # inject the trigger and target pattern

            #self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx:beg_idx + trigger_len] = triggers.detach()
            #self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx + trigger_len:beg_idx + trigger_len + pattern_len] = \
            #    self.target_pattern + self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx - 1:beg_idx]

            # inject the trigger and target pattern
            self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx:beg_idx + trigger_len] = triggers.detach()

            ptn = self.target_pattern
            if ptn.dim() == 2:
                ptn = ptn.unsqueeze(1)  # (n_atk, 1, L)
            self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx + trigger_len:beg_idx + trigger_len + pattern_len] = \
                ptn + self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx - 1:beg_idx]

    def predict_trigger(self, data_bef_trigger):
        """
        predict the trigger using the trigger generator.
        n = number of samples, c = number of variables, l = length of the data
        :param data_bef_trigger: the data before the trigger, shape: (n, c, l).
        :return: the predicted trigger, shape: (n, c, trigger_len)
        """
        c, l = data_bef_trigger.shape[-2:]
        data_bef_trigger = self.dataset.normalize(data_bef_trigger)
        data_bef_trigger = data_bef_trigger.view(-1, self.trigger_generator.input_dim)
        triggers, perturbations = self.trigger_generator(data_bef_trigger)
        triggers = self.dataset.denormalize(triggers).reshape(-1, c, self.trigger_len)
        return triggers, perturbations

    def get_trigger_slices(self, bef_len, aft_len):
        """
        Limit the range for soft identification.
        Find all sliced windows that contain the trigger.
        Time range per slice: [idx - bef_len, idx + aft_len),
        where idx is the *start index* of triggers.

        In addition, for each slice we overwrite the first `pattern_len` steps
        *after* the trigger with the target pattern (per-attack-variable),
        i.e., x[t] = base + pattern, where base is the value at (idx - 1).
        """
        import torch

        slices = []
        timestamps = []

        T = self.dataset.poisoned_data.shape[-1]
        n_atk = int(self.atk_vars.shape[0])


        fct_input_len = bef_len + self.trigger_len
        pat_len = int(self.pattern_len)

        for idx in self.atk_ts.tolist():
            left = idx - bef_len
            right = idx + aft_len

            if right < T and left >= 0:

                slc = self.dataset.poisoned_data[..., left:right].detach().clone()


                max_writable = slc.shape[-1] - fct_input_len
                if max_writable > 0 and (bef_len - 1) >= 0:
                    L = min(pat_len, max_writable)


                    base = slc[self.atk_vars, :, bef_len - 1]  # (n_atk, 1)
                    base = base.unsqueeze(-1)  # (n_atk, 1, 1)


                    ptn = self.target_pattern
                    if not torch.is_tensor(ptn):
                        ptn = torch.as_tensor(ptn)
                    ptn = ptn.to(device=slc.device, dtype=slc.dtype)

                    if ptn.dim() == 1:
                        # (L,) -> (1,1,L) -> (n_atk,1,L)
                        ptn = ptn.view(1, 1, -1).expand(n_atk, 1, -1)
                    elif ptn.dim() == 2:
                        # (n_atk, L) -> (n_atk,1,L)
                        assert ptn.shape[0] == n_atk, \
                            f"target_pattern The first dimension should be n_atk={n_atk}，But get {ptn.shape[0]}"
                        ptn = ptn.unsqueeze(1)
                    elif ptn.dim() == 3:

                        if ptn.shape[0] == 1 and ptn.shape[1] == n_atk:
                            ptn = ptn.permute(1, 0, 2)  # -> (n_atk,1,L)
                        assert ptn.shape[0] == n_atk and ptn.shape[1] == 1, \
                            f"target_pattern should be (n_atk,1,L)，But get {tuple(ptn.shape)}"
                    else:
                        raise ValueError(f"Unexpected target_pattern ndim={ptn.dim()}")

                    # If the length of the target mode is inconsistent with pat_1en, truncate it according to the writable length L
                    if ptn.shape[-1] < L:
                        L = ptn.shape[-1]

                    start = fct_input_len
                    end = start + L
                    slc[self.atk_vars, :, start:end] = base + ptn[..., :L]

                slices.append(slc)

                if self.dataset.use_timestamp:

                    timestamps.append(self.dataset.timestamps[left:right])

        if not self.dataset.use_timestamp:
            return slices
        return slices, timestamps

    def select_atk_timestamp(self, poison_metrics, min_distance=10, alpha=0.5):
        """
        Use weighted causality score and loss score to jointly select the attack time point
        """
        select_pos_mark = torch.zeros(len(self.dataset), dtype=torch.int)
        poison_metrics = torch.cat(poison_metrics, dim=0).to(self.device)

        #Obtain mse/los score
        mse_score = poison_metrics[:, 0]
        mse_score = (mse_score - mse_score.min()) / (mse_score.max() - mse_score.min() + 1e-8)


        causal_score = self.get_deep_causal_scores(
            window_size=self.causal_window_size,
            granger_maxlag=self.causal_granger_maxlag,
            hybrid=self.causal_fusion,
            w_neural=self.causal_w_neural,
            gate_bias=self.causal_gate_bias
        )
        causal_score = (causal_score - causal_score.min()) / (causal_score.max() - causal_score.min() + 1e-8)

        indices = poison_metrics[:, 1].long()
        # Two stages: first make candidates based on causal strength, and then sort them by MAE among the candidates
        k = int(min(len(indices), max(self.temporal_poison_num * 4, self.temporal_poison_num)))
        cand_by_causal = torch.argsort(causal_score[indices], descending=True)[:k]
        cand_mae = mse_score[cand_by_causal]
        sort_idx = cand_by_causal[torch.argsort(cand_mae, descending=True)].detach().cpu().numpy()
        valid_idx = []

        for i in range(len(sort_idx)):
            beg_idx = int(poison_metrics[sort_idx[i], 1])
            end_idx = beg_idx + self.trigger_len + self.pattern_len + 8
            if torch.sum(select_pos_mark[beg_idx:end_idx]) == 0 and \
                    end_idx < len(self.dataset) and beg_idx > self.bef_tgr_len:
                valid_idx.append(sort_idx[i])
                select_pos_mark[beg_idx:end_idx] = 1
            if len(valid_idx) > 2 * self.temporal_poison_num:
                break

        valid_idx = np.array(valid_idx)
        top_sort_idx = np.random.choice(valid_idx, min(self.temporal_poison_num, valid_idx.shape[0]), replace=False)
        top_sort_idx = torch.from_numpy(top_sort_idx).to(self.device)
        atk_ts = poison_metrics[top_sort_idx, 1].long()
        atk_ts = torch.sort(atk_ts)[0]
        self.set_atk_timestamp(atk_ts)
        self.adjust_trigger_distance(min_distance)

    def update_trigger_generator(self, net, epoch, epochs, use_timestamps=False, min_distance=10):
        """
        Update the trigger generator using the soft identification.
        """
        if not use_timestamps:
            tgr_slices = self.get_trigger_slices(self.fct_input_len - self.trigger_len,
                                                 self.trigger_len + self.pattern_len + self.fct_output_len)
        else:
            tgr_slices, tgr_timestamps = self.get_trigger_slices(self.fct_input_len - self.trigger_len,
                                                             self.trigger_len + self.pattern_len + self.fct_output_len)
        pbar = tqdm.tqdm(tgr_slices, desc=f'Attacking data {epoch}/{epochs}')
        for slice_id, slice in enumerate(pbar):
            slice = slice.to(self.device)
            slice = slice[:, 0:1, :]
            n, c, l = slice.shape
            data_bef = slice[self.atk_vars, :,
                       self.fct_input_len - self.trigger_len - self.bef_tgr_len:self.fct_input_len - self.trigger_len]
            data_bef = data_bef.reshape(-1, self.bef_tgr_len)

            triggers, perturbations = self.predict_trigger(data_bef)

            # Add the trigger to the slice. x[t-trigger_len:x] = trigger
            triggers = triggers.reshape(self.atk_vars.shape[0], -1, self.trigger_len)
            slice[self.atk_vars, :, self.fct_input_len - self.trigger_len:self.fct_input_len] = triggers

            # Add the pattern to the slice. x[t:t+ptn_len] = x[t-1-trigger_len] + target_pattern
            #slice[self.atk_vars, :, self.fct_input_len:self.fct_input_len + self.pattern_len] = \
                #self.target_pattern + slice[self.atk_vars, :, self.fct_input_len - self.trigger_len - 1].unsqueeze(-1)


            # Add the pattern to the slice. x[t:t+ptn_len] = x[t-1-trigger_len] + target_pattern
            n_atk = int(self.atk_vars.shape[0])
            start = self.fct_input_len
            end = start + self.pattern_len


            base = slice[self.atk_vars, :, self.fct_input_len - self.trigger_len - 1]  # (n_atk, 1)
            base = base.unsqueeze(-1)  # (n_atk, 1, 1)


            ptn = self.target_pattern
            if not torch.is_tensor(ptn):
                ptn = torch.as_tensor(ptn)
            ptn = ptn.to(device=slice.device, dtype=slice.dtype)

            if ptn.dim() == 1:

                ptn = ptn.view(1, 1, -1).expand(n_atk, 1, -1)
            elif ptn.dim() == 2:

                assert ptn.shape[0] == n_atk, f"target_pattern should be n_atk={n_atk}，but {ptn.shape[0]}"
                ptn = ptn.unsqueeze(1)
            elif ptn.dim() == 3:

                if ptn.shape[0] == 1 and ptn.shape[1] == n_atk:
                    ptn = ptn.permute(1, 0, 2)  # -> (n_atk,1,L)
                assert ptn.shape[0] == n_atk and ptn.shape[
                    1] == 1, f"target_pattern  (n_atk,1,L)， {tuple(ptn.shape)}"
            else:
                raise ValueError(f"Unexpected target_pattern ndim={ptn.dim()}")


            assert ptn.shape[
                       -1] == self.pattern_len, f"pattern_len={self.pattern_len}  target_pattern  {ptn.shape[-1]} "


            slice[self.atk_vars, :, start:end] = base + ptn

            # Mimic the soft identification, i.e., the input and output only contain a part of the trigger and pattern
            batch_inputs_bkd = [slice[..., i:i + self.fct_input_len] for i in range(self.pattern_len)]
            batch_labels_bkd = [slice[..., i + self.fct_input_len:i + self.fct_input_len + self.fct_output_len].detach()
                                for i in range(self.pattern_len)]
            batch_inputs_bkd = torch.stack(batch_inputs_bkd, dim=0)
            batch_labels_bkd = torch.stack(batch_labels_bkd, dim=0)

            batch_inputs_bkd = batch_inputs_bkd[:, :, 0:1, :]
            batch_labels_bkd = batch_labels_bkd[:, :, 0, :]
            batch_inputs_bkd = self.dataset.normalize(batch_inputs_bkd)

            # Calculate eta in the soft identification to reweight the loss
            loss_decay = (self.pattern_len - torch.arange(0, self.pattern_len, dtype=torch.float32).to(
                self.device)) / self.pattern_len

            self.attack_optim.zero_grad()
            batch_inputs_bkd = batch_inputs_bkd.squeeze(2).permute(0, 2, 1)
            batch_labels_bkd = batch_labels_bkd.permute(0, 2, 1)

            if use_timestamps:
                batch_x_mark = [tgr_timestamps[slice_id][i:i + self.fct_input_len] for i in range(self.pattern_len)]
                batch_y_mark = [
                    tgr_timestamps[slice_id][i + self.fct_input_len:i + self.fct_input_len + self.fct_output_len] for i
                    in range(self.pattern_len)]
                batch_x_mark = torch.stack(batch_x_mark, dim=0)
                batch_y_mark = torch.stack(batch_y_mark, dim=0)
            else:
                batch_x_mark = torch.zeros(batch_inputs_bkd.shape[0], batch_inputs_bkd.shape[1], 4).to(self.device)

            x_des = torch.zeros_like(batch_labels_bkd)
            outputs_bkd = net(batch_inputs_bkd, batch_x_mark, x_des, None)
            outputs_bkd = self.dataset.denormalize(outputs_bkd)

            loss_bkd = F.mse_loss(outputs_bkd[:, :, self.atk_vars], batch_labels_bkd[:, :, self.atk_vars],
                                  reduction='none')
            loss_bkd = torch.mean(loss_bkd, dim=(1, 2))
            loss_bkd = torch.sum(loss_bkd * loss_decay)  # reweight the loss
            loss_norm = torch.abs(torch.sum(perturbations, dim=1)).mean()
            loss = loss_bkd + self.lam_norm * loss_norm

            loss.backward()
            self.attack_optim.step()
        self.atk_scheduler.step()

        # Adjust trigger distance
        self.adjust_trigger_distance(min_distance)

    def adjust_trigger_distance(self, min_distance):
        """
        Dynamically adjust the trigger timestamps to ensure a minimum distance between them.
        """
        assert hasattr(self, 'atk_ts'), 'Please set the attack timestamp first.'
        atk_ts = self.atk_ts.cpu().detach().numpy()
        sorted_atk_ts = np.sort(atk_ts)
        adjusted_atk_ts = [sorted_atk_ts[0]]

        for idx in sorted_atk_ts[1:]:
            if idx - adjusted_atk_ts[-1] >= min_distance:
                adjusted_atk_ts.append(idx)
            else:
                # Adjust the current timestamp to maintain the minimum distance
                adjusted_atk_ts.append(adjusted_atk_ts[-1] + min_distance)

        # Ensure the number of triggers remains the same
        adjusted_atk_ts = adjusted_atk_ts[:len(atk_ts)]
        adjusted_atk_ts = np.sort(adjusted_atk_ts)
        self.set_atk_timestamp(torch.tensor(adjusted_atk_ts).long().to(self.device))

    def get_deep_causal_scores(self,
                               window_size: int = 12,
                               granger_maxlag: int = 2,
                               hybrid: str = "weighted",
                               w_neural: float = 0.8,
                               gate_bias: float = 0.2,
                               pkey: str = "ssr_ftest",
                               granger_stride: int = 72,  # <== 新增：每隔多少步做一次 Granger（这里144=12小时）72=6
                               granger_topk: int = 15  # <== 新增：只对神经因果得分 TopK 的父节点做 Granger
                               ):
        """
        Calculate the "Neurocausality Score" and "Granger Significance Score",
        and fuse them into the final causal strength using * double weighting/gating *.
        """
        import numpy as np
        import torch
        from statsmodels.tsa.stattools import grangercausalitytests


        data = self.dataset.data.detach().cpu()
        n, c, T = data.shape

        neural_scores = torch.zeros(T, dtype=torch.float32)
        granger_scores = torch.zeros(T, dtype=torch.float32)


        atk_var = self.atk_vars[0].item() if isinstance(self.atk_vars, torch.Tensor) else self.atk_vars[0]


        adjs = []

        for t in range(window_size, T, granger_stride):

            t_next = min(T, t + granger_stride)
            neural_scores[t:t_next] = neural_scores[t]
            granger_scores[t:t_next] = granger_scores[t]

            window = data[..., t - window_size:t]  # [n, c, w]

            context_vec_flat = window.mean(dim=-1).flatten().unsqueeze(0)  # [1, n*c]
            adj = self.causal_module(context_vec_flat.to(self.device))  # [1, n, n]
            neural_score = adj[0, :, atk_var].mean().detach().cpu().float()
            neural_scores[t] = neural_score


            adjs.append(adj[0].detach().cpu())


            y = window[atk_var].mean(dim=0).numpy()
            maxlag = int(min(granger_maxlag, max(1, window_size - 2)))

            sigs = []


            cand_vals = adj[0, :, atk_var].detach().cpu()
            cand_vals[atk_var] = -1e9
            topk = min(int(granger_topk), cand_vals.numel() - 1)
            cand_idx = torch.topk(cand_vals, k=topk, largest=True).indices.cpu().numpy().tolist()

            for j in cand_idx:
                ...

                x = window[j].mean(dim=0).numpy()

                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    continue
                if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                    continue
                try:
                    arr = np.column_stack([y, x])
                    res = grangercausalitytests(arr, maxlag=maxlag, verbose=False)
                    pvals = []
                    for lag, out in res.items():
                        test = out[0]
                        if pkey in test and test[pkey][1] is not None and np.isfinite(test[pkey][1]):
                            pvals.append(float(test[pkey][1]))
                        else:
                            for alt in ("ssr_ftest", "lrtest", "params_ftest", "ssr_chi2test", "f_test"):
                                if alt in test and test[alt] is not None:
                                    pv = test[alt][1] if isinstance(test[alt], (list, tuple)) else getattr(test[alt],
                                                                                                           "pvalue",
                                                                                                           None)
                                    if pv is not None and np.isfinite(pv):
                                        pvals.append(float(pv))
                                        break
                    if pvals:
                        sigs.append(max(0.0, min(1.0, 1.0 - min(pvals))))
                except Exception:
                    pass
            granger_scores[t] = float(np.mean(sigs)) if len(sigs) > 0 else 0.0


            if len(adjs) > 0:
                A_prior = torch.stack(adjs, dim=0).mean(0)  # [n, n]
                A_prior = A_prior / (A_prior.max() + 1e-8)  # 0~1 归一化，作为强度先验
                self.causal_prior_A = A_prior.to(self.device)

                try:
                    import numpy as np
                    np.save(self.causal_prior_save, self.causal_prior_A.detach().cpu().numpy())
                    print(
                        f"[Attacker] Saved causal prior to {self.causal_prior_save}, shape={tuple(self.causal_prior_A.shape)}")
                except Exception as e:
                    print(f"[Attacker] Save causal prior failed: {e}")


                if hasattr(self.trigger_generator, "set_causal_prior"):
                    self.trigger_generator.set_causal_prior(self.causal_prior_A)


        def minmax_norm(x: torch.Tensor):
            seg = x[window_size:]
            if seg.numel() == 0:
                return x
            mn, mx = torch.min(seg), torch.max(seg)
            if float(mx - mn) <= 1e-12:
                x[window_size:] = 0.0
            else:
                x[window_size:] = (seg - mn) / (mx - mn + 1e-8)
            return x

        neural_scores = minmax_norm(neural_scores)
        granger_scores = minmax_norm(granger_scores)


        if hybrid == "gated":
            combined = neural_scores * (gate_bias + (1.0 - gate_bias) * granger_scores)
        else:
            w = float(np.clip(w_neural, 0.0, 1.0))
            combined = w * neural_scores + (1.0 - w) * granger_scores

        combined = minmax_norm(combined)
        return combined.to(self.device)
