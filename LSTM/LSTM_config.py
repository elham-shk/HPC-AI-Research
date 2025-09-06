import itertools
import random

# Define hyperparameter search space
# You can narrow these down later if needed
num_layers_list         = [4, 5, 6,7,8]                          # Number of LSTM layers  #1 [1, 2, 3, 4, 5, 6]
#units_list              = [32, 64, 128, 256, 512]                # Units per LSTM layer
units_list = [
    # 1 layer
    [128],
    [256],

    # 2 layers
    [128, 256],  # expanding
    [256, 128],  # tapering

    # 3 layers
    [128, 256, 512],
    [512, 256, 128],
    [128, 128, 128],  # flat

    # 4 layers
    [64, 128, 256, 512],
    [512, 256, 128, 64],
    [128, 128, 128, 128],  # flat

    # 5 layers
    [64, 128, 256, 512, 512],
    [512, 512, 256, 128, 64],
    [128, 128, 128, 128, 128],

    # 6 layers
    [64, 128, 256, 256, 512, 512],
    [512, 512, 256, 256, 128, 64],
    [128, 128, 128, 128, 128, 128],

    # 7 layers
    [64, 128, 256, 256, 512, 512, 512],
    [512, 512, 256, 256, 128, 64, 32],

    # 8 layers
    [32, 64, 128, 256, 256, 512, 512, 512],
    [512, 512, 256, 256, 128, 128, 64, 32],

    # 9 layers
    [32, 64, 128, 128, 256, 256, 512, 512, 512],
    [512, 512, 256, 256, 128, 128, 64, 64, 32],

    # 10 layers
    [32, 64, 64, 128, 128, 256, 256, 512, 512, 512],
    [512, 512, 512, 256, 256, 128, 128, 64, 64, 32],
    [128] * 10
]

dropout_list            = [0.0, 0.2, 0.4]                        # Dropout after LSTM
recurrent_dropout_list  = [0.0, 0.2, 0.4]                        # Recurrent dropout in LSTM
batch_size_list         = [64, 128, 256]                         # Batch sizes
dense_units_list        = [64, 128, 256, 512]                    # Units in hidden dense layer
dense_activation_list   = ['relu']                              # Hidden dense layer activation (fixed)
dense_dropout_list      = [0.0, 0.2, 0.3]
num_dense_layers_list   = [1, 2, 3]                        # Dropout after dense

# Create configuration combinations
lstm_configs = []
for combo in itertools.product(
   # num_layers_list,
    units_list,
    dropout_list,
    recurrent_dropout_list,
    batch_size_list,
    dense_units_list,
    dense_activation_list,
    dense_dropout_list,
    num_dense_layers_list
):
    cfg = {
    #    "num_layers": combo[0],
        "units": combo[0],
        "num_layers": len(combo[0]),
        "dropout": combo[1],
        "recurrent_dropout": combo[2],
        "batch_size": combo[3],
        "dense_units": combo[4],
        "dense_activation": combo[5],  # always 'relu'
        "dense_dropout": combo[6],
        "num_dense_layers": combo[7]
    }
    lstm_configs.append(cfg)

# Shuffle and optionally limit
print(f"Total valid configurations before shuffling: {len(lstm_configs)}")
random.seed(42)
random.shuffle(lstm_configs)
print(f"Total valid configurations after shuffling: {len(lstm_configs)}")

# Sample subset (optional)
sampled_configs = lstm_configs[:6000]
print(f"Sampled {len(sampled_configs)} configurations.")

lstm_configs=sampled_configs
