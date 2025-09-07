
import sys
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os
import tensorflow.keras.backend as K
from LSTM_config import lstm_configs
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import sys
from sklearn.metrics import r2_score

seed = 42
np.random.seed(seed)





###############################################################################

import ast

import os, json, hashlib, ast


#def config_hash(cfg):
#    """
#    Canonical MD5 hash of a hyper‑param dict.
#    """
#    s = json.dumps(cfg, sort_keys=True)
#    return hashlib.md5(s.encode("utf-8")).hexdigest()


def normalize_config(cfg):
    normalized = {}
    for k, v in cfg.items():
        if isinstance(v, float):
            # Round floats 
            normalized[k] = round(v, 6)
        elif isinstance(v, list):
            # Normalize each element in list
            normalized[k] = [
                round(i, 6) if isinstance(i, float) else i for i in v
            ]
        elif isinstance(v, str):
            # Strip any whitespace from strings
            normalized[k] = v.strip()
        else:
            normalized[k] = v
    return normalized

def config_hash(cfg):
    normalized = normalize_config(cfg)
    s = json.dumps(normalized, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()




def load_tested_configs(file_path):
    """
    Parse results/tcn_metrics.txt,
    split on “=== Experiment” and turn each block
    back into a dict of hyper‑parameters.
    """
    if not os.path.exists(file_path):
        return []
    text = open(file_path).read()
    blocks = text.split("=== Experiment")[1:]
    tested = []
    for blk in blocks:
        params = {}
        for line in blk.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                # convert to int, float, list, or bool
                v = ast.literal_eval(v)
            except Exception:
                # Fallback to stripped string
                v = v.strip()
            params[k] = v
        tested.append(params)
    return tested


def load_tested_hashes(file_path):
    tested_hashes = set()
    tested_configs = load_tested_configs(file_path)
    for cfg in tested_configs:
        tested_hashes.add(config_hash(cfg))
    return tested_hashes



##########################################################################


SUMMARY       = os.path.join("results", "lstm_metrics_all.txt")
tested_list   = load_tested_configs(SUMMARY)
tested_hashes = {config_hash(cfg) for cfg in tested_list}
print(f"[INFO] {len(tested_hashes)} hyper‑parameter sets already tested")



######################################################################


# === Pick config from command-line argument ===
if len(sys.argv) != 2:
    print("Usage: python run_lstm_model.py <config_index>", file=sys.stderr)
    sys.exit(1)

idx = int(sys.argv[1])
if idx < 0 or idx >= len(lstm_configs):
    print(f"Config index must be in [0..{len(lstm_configs)-1}]", file=sys.stderr)
    sys.exit(1)

cfg = lstm_configs[idx]
print(f"\n=== Running LSTM Experiment {idx}/{len(lstm_configs)-1} ===")
print(cfg)





# Load data
train = np.load('../train.npy', allow_pickle=True).item()
test = np.load('../test.npy', allow_pickle=True).item()
train_x, train_y = train['X'], train['Y']
test_x, test_y = test['X'], test['Y']


# Normalize input
mean = train_x.mean(axis=0, keepdims=True)
std = train_x.std(axis=0, keepdims=True) + 1e-6
train_x = (train_x - mean)
test_x = (test_x - mean)



#######################Model###############

def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_lstm_model(cfg, input_shape, output_dim):
    model = Sequential()

    num_layers = len(cfg["units"])


    for layer_idx in range(num_layers):

        return_seq = layer_idx <num_layers - 1
        units = cfg["units"][layer_idx]  # pick from list

        if layer_idx == 0:
            model.add(LSTM(
               # units=cfg["units"],
                units=units,
                return_sequences=return_seq,
                dropout=cfg["dropout"],
                recurrent_dropout=cfg["recurrent_dropout"],
                input_shape=input_shape
            ))
        else:
             model.add(LSTM(
               # units=cfg["units"],
                 units=units,
                 return_sequences=return_seq,
                 dropout=cfg["dropout"],
                 recurrent_dropout=cfg["recurrent_dropout"]
            ))

    for _ in range(cfg["num_dense_layers"]):
        model.add(Dense(cfg["dense_units"], activation=cfg["dense_activation"]))
        model.add(Dropout(cfg["dense_dropout"]))



    # Output layer
    model.add(Dense(output_dim, activation='linear'))

    # Compile with RMSE loss
    model.compile(optimizer='adam', loss=rmse_loss, metrics=['mae'])
    return model


######################
# 6) instantiate & train

# ─── Define the time‑limit callback ───────────────────

class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_time_per_epoch=600):
        super().__init__()
        self.max_time = max_time_per_epoch
        self.timed_out = False

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

#    def on_epoch_end(self, epoch, logs=None):
#        epoch_duration = time.time() - self.epoch_start
#        if epoch_duration > self.max_time:
#            print(f"\n[TimeLimit] Epoch {epoch} took {epoch_duration:.1f}s > {self.max_time}s—stopping.")
#            self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        elapsed = time.time() - self.epoch_start

        if elapsed > self.max_time:
            print(f"\n[TimeLimit] Batch {batch} pushed epoch time to {elapsed:.1f}s > {self.max_time}s → stopping.")
            self.model.stop_training = True
            self.timed_out = True 

time_cb = TimeLimitCallback(max_time_per_epoch=3000)



#------ If exeecds job walltime__________________________________________________

import time
import tensorflow as tf

# ── callback to stop training when total job time exceeds a threshold ──
class WalltimeCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_time, max_seconds, margin=60):
        """
        start_time: timestamp when job/script began
        max_seconds: total allowed run (e.g. 11*3600 for 11h)
        margin: seconds left to leave as buffer (default 1 min)
        """
        super().__init__()
        self.start_time = start_time
        self.max_seconds = max_seconds
        self.margin = margin
        self.timed_out = False

    def on_train_batch_end(self, batch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed + self.margin > self.max_seconds:
            print(f"\n[WalltimeCallback] hit {elapsed:.0f}s > {self.max_seconds}s — stopping now.")
            self.model.stop_training = True
            self.timed_out = True


# record when we began (right before build/train)
job_start = time.time()
MAX_RUN = 11 * 3600    # 11 hours in seconds
wall_cb = WalltimeCallback(start_time=job_start, max_seconds=MAX_RUN)




####################################################


input_shape = (train_x.shape[1], train_x.shape[2])
output_dim = train_y.shape[1]

model = build_lstm_model(cfg, input_shape, output_dim)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    time_cb,
    wall_cb,
]

history = model.fit(
    train_x, train_y,
    validation_split=0.1,
    epochs=500,
    batch_size=cfg["batch_size"],
    callbacks=callbacks,
    verbose=1,
    shuffle=False
)


################ Predict
predicted_y = model.predict(test_x)
mse_val = mean_squared_error(test_y, predicted_y)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(test_y, predicted_y)
#r2_val = r2_score(test_y, predicted_y)
pearson_corr = pearsonr(test_y.flatten(), predicted_y.flatten())[0]

#pearson_corr = np.corrcoef(predicted_y_combined.flatten(), test_y.flatten())[0,1]
r2_val = r2_score(test_y.reshape(-1), predicted_y.reshape(-1))


# === Save results ===
os.makedirs("results", exist_ok=True)



results = {
    "num_layers": len(cfg["units"]),
#   "units": cfg.get("units", lstm_configs[0]["units"]),
    "units": list(cfg.get("units", lstm_configs[0]["units"])),
    "dropout": cfg.get("dropout", lstm_configs[0]["dropout"]),
    "recurrent_dropout": cfg.get("recurrent_dropout", lstm_configs[0]["recurrent_dropout"]),
    "batch_size": cfg.get("batch_size", lstm_configs[0]["batch_size"]),
    "dense_units": cfg.get("dense_units", lstm_configs[0]["dense_units"]),
    "dense_activation": cfg.get("dense_activation", lstm_configs[0]["dense_activation"]),
    "dense_dropout": cfg.get("dense_dropout", lstm_configs[0]["dense_dropout"]),
    "num_dense_layers": cfg.get("num_dense_layers", lstm_configs[0]["num_dense_layers"]),


    # Metrics
    "mse": mse_val,
    "rmse": rmse_val,
    "mae": mae_val,
    "r2": r2_val,
    "pearson": pearson_corr
}

summary_path = f"results/lstm_metrics.txt"

#with open(summary_path, "a") as f:
#    f.write(f"=== Experiment {idx} ===\n")
#    for key, val in results.items():
#        f.write(f"{key}: {val}\n")
#    f.write("\n")


with open(summary_path, "a") as f:
    if time_cb.timed_out:
     
        # Write the config but mark it rejected
        f.write(f"=== Experiment {idx} REJECTED ===\n")
        for key, val in config.items():
            f.write(f"{key}: {val}\n")
        f.write("\n")
    else:
        # Your existing code to write out the results dict
        f.write(f"=== Experiment {idx} ===\n")
        for key, val in results.items():
            f.write(f"{key}: {val}\n")
        f.write("\n")

print(f"Experiment {idx} {'rejected' if time_cb.timed_out else 'complete'}")




# === Save figure ===
plt.figure()
plt.scatter(test_y.flatten(), predicted_y.flatten(), alpha=0.3)
plt.plot([min(test_y.flatten()), max(test_y.flatten())],
         [min(test_y.flatten()), max(test_y.flatten())], 'k--')
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title(f"Observed vs Predicted (Config {idx})")
plt.grid(True)
#plt.savefig(f"results/fig_exp{idx}.png")
plt.close()

print(f"Experiment {idx} complete. Summary and figure saved in results/")
