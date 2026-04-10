# MTBTime
<a name="configuration"></a>
### 2. Configuration
- **`configs/train_config.yaml`**: Configuration file for the project. Specify the hyperparameters and training details.
- **`configs/default_config.yaml`**: Configuration file for the dataset/model/target pattern.

[//]: # (insert a tip to modify the configuration file)
Tips: Feel free to modify `model_name`, `dataset`, `pattern_type` in `train_config.yaml` file:
```yaml
Train:
    batch_size: 64
    learning_rate: 0.0001
    attack_lr: 0.005
    num_epochs: 50
    warmup: 10
    gpuid: '0'
    surrogate_name: FEDformer
    model_name: FEDformer
    dataset: PEMS03
    pattern_type: [ cone, up_trend ]
    trigger_len: 4
    pattern_len: 7
    bef_tgr_len: 6
    lam_norm: 0.05
    alpha_s: 0.3  # spatial_poison_rate
    alpha_t: 0.03  # temporal_poison_rate
    epsilon: 0.2
    hidden_dim: 64  # hidden_dim for the trigger generator
