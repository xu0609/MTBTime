# 文件 3：run.py
import torch
import numpy as np
import os
import random
from dataset import load_raw_data
from trainer import Trainer
import yaml
from easydict import EasyDict as edict

def seed_torch(seed=1) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parser_args():
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)
    config = yaml.load(open('configs/train_config.yaml'), Loader=yaml.FullLoader)['Train']
    config['Dataset'] = default_config['Dataset'][config['dataset']]
    #config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]
    pt = config['pattern_type']
    # 支持多目标模式：pattern_type 可以是字符串、逗号分隔字符串，或列表
    if isinstance(pt, list):
        config['Target_Pattern'] = [default_config['Target_Pattern'][p] for p in pt]
    elif isinstance(pt, str) and (',' in pt):
        pt_list = [p.strip() for p in pt.split(',') if p.strip()]
        config['Target_Pattern'] = [default_config['Target_Pattern'][p] for p in pt_list]
    else:
        config['Target_Pattern'] = default_config['Target_Pattern'][pt]
    config['Model'] = default_config['Model'][config['model_name']]
    config['Model']['c_out'] = config['Dataset']['num_of_vertices']
    config['Model']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Model']['dec_in'] = config['Dataset']['num_of_vertices']
    config['Surrogate'] = default_config['Model'][config['surrogate_name']]
    config['Surrogate']['c_out'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['dec_in'] = config['Dataset']['num_of_vertices']
    config = edict(config)
    return config


def main(config):
    gpuid = config.gpuid
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print("CUDA:", USE_CUDA, DEVICE)

    seed_torch()

    data_config = config.Dataset
    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(data_config)
        train_data_stamps = test_data_stamps = None
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(data_config)

    spatial_poison_num = max(int(round(train_data_seq.shape[1] * config.alpha_s)), 1)
    atk_vars = np.arange(train_data_seq.shape[1])
    atk_vars = np.random.choice(atk_vars, size=spatial_poison_num, replace=False)
    atk_vars = torch.from_numpy(atk_vars).long().to(DEVICE)
    print('shape of attacked_variables', atk_vars.shape)

   # target_pattern = config.Target_Pattern
    #target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std
    target_cfg = config.Target_Pattern

    if isinstance(target_cfg, (list, tuple)) and target_cfg and isinstance(target_cfg[0], (list, tuple)):
        import numpy as _np
        _all_patterns = torch.tensor(_np.array(target_cfg)).float().to(DEVICE) * train_std  # (k, L)
        _k = _all_patterns.shape[0]
        _n_atk = atk_vars.shape[0]
        _assign = torch.arange(_n_atk, device=DEVICE) % _k  # 0..k-1
        target_pattern = _all_patterns[_assign]  #
    else:
        target_pattern = torch.tensor(target_cfg).float().to(DEVICE) * train_std


    exp_trainer = Trainer(config, atk_vars, target_pattern, train_mean, train_std, train_data_seq, test_data_seq,
                          train_data_stamps, test_data_stamps, DEVICE)

    save_file = f'./checkpoints/attacker_{config.dataset}.pth'
    if os.path.exists(save_file):
        state = torch.load(save_file)
        exp_trainer.load_attacker(state)
        print('load attacker from', save_file)
    else:
        print('=' * 20, ' [ Stage 1 ] ', '=' * 20)
        print('start training surrogate model and attacker')
        exp_trainer.train()

        state = exp_trainer.save_attacker()
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(state, save_file)

    print('=' * 20, ' [ Stage 2 ] ', '=' * 20)
    print('start evaluating attack performance on a new model')
    exp_trainer.test()

    avg_metrics = exp_trainer.compute_average_metrics()
    print(f"\nAverage Metrics Over All Epochs:")
    print(f"Clean MAE: {avg_metrics['avg_clean_mae']}")
    print(f"Clean RMSE: {avg_metrics['avg_clean_rmse']}")
    print(f"Attacked MAE: {avg_metrics['avg_attacked_mae']}")
    print(f"Attacked RMSE: {avg_metrics['avg_attacked_rmse']}")

if __name__ == "__main__":
    config = parser_args()
    main(config)