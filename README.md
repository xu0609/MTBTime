# MTBTime
<a name="configuration"></a>
### 2. Configuration
- **`configs/train_config.yaml`**: Configuration file for the project. Specify the hyperparameters and training details.
- **`configs/default_config.yaml`**: Configuration file for the dataset/model/target pattern.

[//]: # (insert a tip to modify the configuration file)
Tips: Feel free to modify `model_name`, `dataset`, `pattern_type` in `train_config.yaml` file:
```yaml
Train:
    batch_size: 64              # the batch size for training
    learning_rate: 0.0001       # the learning rate for the surrogate model
    attack_lr: 0.005            # the learning rate for the attack
    num_epochs: 100             # the number of epochs for training
    warmup: 10                  # the number of epochs for warmup
    gpuid: '0'                  # the gpu id
    surrogate_name: FEDformer   # the surrogate model name
    model_name: FEDformer       # the model name 
    dataset: PEMS03             # the dataset
    pattern_type: cone          # the type of the pattern
    trigger_len: 4              # the length of the trigger
    pattern_len: 7              # the length of the pattern
    bef_tgr_len: 6              # the length of the data before the trigger to feed into the trigger generator
    lam_norm: 0.05              # the weight for the norm loss
    alpha_s: 0.3                # spatial_poison_rate
    alpha_t: 0.03               # temporal_poison_rate
    epsilon: 0.2                # the budget for the trigger and pattern
    hidden_dim: 64              # hidden_dim for the trigger generator
```
