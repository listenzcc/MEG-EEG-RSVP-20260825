
'''
Require python3.11 braindecode venv
'''

# %%
from skorch.dataset import ValidSplit
from torch.utils.data import TensorDataset
import mne
import torch

from braindecode.models import EEGNetv4
from braindecode import EEGClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from skorch.callbacks import (
    Callback,
    EarlyStopping,
    LRScheduler,
)

from util.easy_imports import *


# %%
parser = argparse.ArgumentParser(
    description='Require SUBJ and MODE parameters')
parser.add_argument('-s', '--subject', default='S02',
                    help='Subject name like S02')
parser.add_argument('-m', '--mode', default='EEG',
                    help='Mode name EEG | MEG')
parser.add_argument('-a', '--flag-remove-artificial',
                    action='store_true', help='using the remove-artificial-epochs')
parser.add_argument('-d', '--decoding_method', default='EEGNetV4',
                    help='Decoding method like EEGNetV4')

args = parser.parse_args()
SUBJ = args.subject
MODE = args.mode
FLAG_REMOVE_ARTIFICIAL = args.flag_remove_artificial
DECODING_METHOD = args.decoding_method

if FLAG_REMOVE_ARTIFICIAL:
    DECODING_METHOD += '-RA'

print(args)

# %%
DATA_DIR = Path(f'output/epochs/{MODE}-{SUBJ}')

OUTPUT_DIR = Path(
    f'output/neuralnetwork-decode/{MODE}-{SUBJ}-{DECODING_METHOD}')

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
# Target (1) vs Non-target (2)
if FLAG_REMOVE_ARTIFICIAL:
    epochs_1 = mne.read_epochs(
        DATA_DIR / 'epochs-1-notch-removal-artificial-epo.fif')
    epochs_2 = mne.read_epochs(
        DATA_DIR / 'epochs-2-notch-removal-artificial-epo.fif')
else:
    epochs_1 = mne.read_epochs(DATA_DIR / 'epochs-1-notch-epo.fif')
    epochs_2 = mne.read_epochs(DATA_DIR / 'epochs-2-notch-epo.fif')

# epochs_3 = mne.read_epochs(DATA_DIR / 'epochs-3-notch-epo.fif')
epochs_all = mne.concatenate_epochs([epochs_1, epochs_2])
print(epochs_all)

# 仅保留刺激后0-0.6秒
epochs_all = epochs_all.crop(tmin=0.0, tmax=0.6)

# 获取数据X和标签y
X = epochs_all.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs_all.events[:, -1]  # 标签 1,2,3,4 ...
y[y != 1] = 0  # 将非目标标签设为0

n_epochs, n_channels, n_samples = X.shape

# %%
print()
print('Data shape:')
print(f'  n_epochs   = {n_epochs}')
print(f'  n_channels = {n_channels}')
print(f'  n_samples  = {n_samples}')

print()
print('Class distribution:')
print(f'  non-target (0): {(y == 0).sum()}')
print(f'  target     (1): {(y == 1).sum()}')


# %%
# ============================================================
# Convert to float32
# ============================================================

X = X.astype(np.float32)
y = y.astype(np.int64)


# %%
# ============================================================
# Check data
# ============================================================

assert X.ndim == 3

assert X.shape[0] == y.shape[0]

assert np.isfinite(X).all(), \
    'X contains NaN or Inf'

assert set(np.unique(y)).issubset({0, 1}), \
    f'Unexpected labels: {np.unique(y)}'


# %%
# ============================================================
# Train / validation / test split
#
# 70% train
# 15% validation
# 15% test
#
# Stratified split keeps target/non-target ratio.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    # random_state=42,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    # 0.17647 * 0.85 ~= 0.15
    stratify=y_train,
    # random_state=42,
)

print()
print('Split:')
print(f'  train: {X_train.shape}, {y_train.shape}')
print(f'  valid: {X_valid.shape}, {y_valid.shape}')
print(f'  test : {X_test.shape}, {y_test.shape}')


# %%
# ============================================================
# EEG normalization
#
# Important:
# Calculate normalization parameters ONLY from training data.
#
# We normalize each channel using training-set mean/std.
# ============================================================

train_mean = X_train.mean(
    axis=(0, 2),
    keepdims=True
)

train_std = X_train.std(
    axis=(0, 2),
    keepdims=True
)

# Prevent division by zero
# train_std[train_std < 1e-6] = 1.0


X_train = (
    X_train - train_mean
) / train_std

X_valid = (
    X_valid - train_mean
) / train_std

X_test = (
    X_test - train_mean
) / train_std


# %%
# ============================================================
# Save normalization parameters
# ============================================================

np.savez(
    OUTPUT_DIR / 'normalization.npz',
    mean=train_mean,
    std=train_std,
)


# %%
# ============================================================
# EEGNetV4
# ============================================================

model = EEGNetv4(
    n_chans=n_channels,
    n_outputs=2,
    n_times=n_samples,

    F1=8,
    D=2,
    F2=16,

    kernel_length=64,

    pool_mode='mean',

    drop_prob=0.25,

    batch_norm_momentum=0.01,

    conv_spatial_max_norm=1.0,
)


print()
print('=' * 70)
print('EEGNetV4')
print('=' * 70)

print(model)


# %%
# ============================================================
# Test forward pass
# ============================================================

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print()
print('Device:', device)


model = model.to(device)

with torch.no_grad():

    dummy = torch.tensor(
        X_train[:4],
        dtype=torch.float32,
        device=device,
    )

    dummy_output = model(dummy)

print()
print('Forward test:')
print('input :', dummy.shape)
print('output:', dummy_output.shape)

assert dummy_output.shape == (4, 2)


# %%
# Move model back to CPU.
#
# EEGClassifier will handle device itself.
# ============================================================

model = model.cpu()


# %%
# ============================================================
# Class weights
#
# RSVP usually has many more non-target epochs than target
# epochs. Weighted CrossEntropyLoss prevents the classifier
# from simply predicting non-target.
# ============================================================

n_non_target = np.sum(y_train == 0)
n_target = np.sum(y_train == 1)

print()
print('Training class distribution:')
print('  non-target:', n_non_target)
print('  target    :', n_target)


# Balanced class weights:
#
# weight_c = N / (K * N_c)
#
# where K = 2

n_train = len(y_train)

weight_non_target = (
    n_train /
    (2.0 * n_non_target)
)

weight_target = (
    n_train /
    (2.0 * n_target)
)

class_weights = torch.tensor(
    [
        weight_non_target,
        weight_target,
    ],
    dtype=torch.float32,
)

print()
print('Class weights:')
print(class_weights)


# %%
# ============================================================
# Callback: print metrics after every epoch
# ============================================================

class PrintEpochMetrics(Callback):

    def on_epoch_end(
        self,
        net,
        dataset_train=None,
        dataset_valid=None,
        **kwargs,
    ):

        history = net.history

        if len(history) == 0:
            return

        row = history[-1]

        epoch = row['epoch']

        train_loss = row.get(
            'train_loss',
            np.nan
        )

        valid_loss = row.get(
            'valid_loss',
            np.nan
        )

        train_acc = row.get(
            'train_accuracy',
            np.nan
        )

        valid_acc = row.get(
            'valid_accuracy',
            np.nan
        )

        print(
            f'Epoch {epoch:03d} | '
            f'train_loss={train_loss:.5f} | '
            f'valid_loss={valid_loss:.5f} | '
            f'train_acc={train_acc:.4f} | '
            f'valid_acc={valid_acc:.4f}'
        )


# %%
# ============================================================
# Train
#
# EEGClassifier needs an explicit validation set.
#
# We use valid data through Dataset-like tuple.
# ============================================================


# Re-create classifier with a validation split based on
# training data.
#
# Since we already created X_valid/y_valid above, it is cleaner
# to concatenate train + valid and let skorch create the split
# only from the training portion.
#
# However, to ensure exactly our predefined validation set,
# we use a custom Dataset below.

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train),
)

valid_dataset = TensorDataset(
    torch.from_numpy(X_valid),
    torch.from_numpy(y_valid),
)


# %%
# ============================================================
# Reinitialize classifier using explicit validation data
# ============================================================

clf = EEGClassifier(

    module=model,

    criterion=torch.nn.CrossEntropyLoss,

    criterion__weight=class_weights,

    optimizer=torch.optim.AdamW,

    optimizer__lr=1e-3,

    optimizer__weight_decay=1e-4,

    batch_size=64,

    max_epochs=100,

    iterator_train__shuffle=True,

    train_split=ValidSplit(
        0.2,
        stratified=True,
        random_state=42,
    ),

    device=device,

    callbacks=[

        PrintEpochMetrics(),

        EarlyStopping(
            monitor='valid_loss',
            patience=15,
        ),

        LRScheduler(
            policy=torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=100,
        ),
    ],
)


# %%
# ============================================================
# Train
# ============================================================

print()
print('=' * 70)
print('Training')
print('=' * 70)

clf.fit(
    X_train,
    y_train,
)


# %%
# ============================================================
# Prediction
# ============================================================

print()
print('=' * 70)
print('Prediction')
print('=' * 70)


y_pred = clf.predict(
    X_test
)

y_prob = clf.predict_proba(
    X_test
)[:, 1]


# %%
# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred,
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    y_prob,
)

cm = confusion_matrix(
    y_test,
    y_pred,
)


# %%
# ============================================================
# Print results
# ============================================================

print()
print('=' * 70)
print('Test results')
print('=' * 70)

print(
    f'Accuracy          : {accuracy:.4f}'
)

print(
    f'Balanced accuracy : {balanced_accuracy:.4f}'
)

print(
    f'ROC-AUC           : {roc_auc:.4f}'
)

print(
    f'F1                : {f1:.4f}'
)

print(
    f'Precision         : {precision:.4f}'
)

print(
    f'Recall            : {recall:.4f}'
)

print()
print('Confusion matrix:')
print(cm)

print()
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            'non-target',
            'target',
        ],
        digits=4,
        zero_division=0,
    )
)


# %%
# ============================================================
# Save predictions
# ============================================================

prediction_file = (
    OUTPUT_DIR /
    'test_predictions.npz'
)

np.savez(
    prediction_file,

    y_true=y_test,

    y_pred=y_pred,

    y_prob=y_prob,
)


# %%
# ============================================================
# Save metrics
# ============================================================

metrics = {
    'subject': SUBJ,
    'mode': MODE,
    'decoding_method': DECODING_METHOD,

    'n_epochs': n_epochs,
    'n_channels': n_channels,
    'n_samples': n_samples,

    'n_train': len(y_train),
    'n_valid': len(y_valid),
    'n_test': len(y_test),

    'n_train_target': int(
        np.sum(y_train == 1)
    ),

    'n_train_non_target': int(
        np.sum(y_train == 0)
    ),

    'n_test_target': int(
        np.sum(y_test == 1)
    ),

    'n_test_non_target': int(
        np.sum(y_test == 0)
    ),

    'accuracy': float(accuracy),

    'balanced_accuracy': float(
        balanced_accuracy
    ),

    'roc_auc': float(
        roc_auc
    ),

    'f1': float(f1),

    'precision': float(precision),

    'recall': float(recall),

    'confusion_matrix': cm.tolist(),
}


with open(
    OUTPUT_DIR / 'metrics.json',
    'w',
    encoding='utf-8',
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# %%
# ============================================================
# Save model
# ============================================================

model_file = (
    OUTPUT_DIR /
    'eegnetv4_model.pt'
)

torch.save(
    clf.module_.state_dict(),
    model_file,
)


# %%
# ============================================================
# Save configuration
# ============================================================

config = {

    'subject': SUBJ,

    'mode': MODE,

    'decoding_method': DECODING_METHOD,

    'remove_artificial':
        FLAG_REMOVE_ARTIFICIAL,

    'n_chans':
        n_channels,

    'n_times':
        n_samples,

    'n_outputs':
        2,

    'F1':
        8,

    'D':
        2,

    'F2':
        16,

    'kernel_length':
        64,

    'pool_mode':
        'mean',

    'drop_prob':
        0.25,

    'learning_rate':
        1e-3,

    'weight_decay':
        1e-4,

    'batch_size':
        64,

    'max_epochs':
        100,
}


with open(
    OUTPUT_DIR / 'config.json',
    'w',
    encoding='utf-8',
) as f:

    json.dump(
        config,
        f,
        indent=4,
    )


# %%
# ============================================================
# Final
# ============================================================

print()
print('=' * 70)
print('Finished')
print('=' * 70)

print(
    f'Output directory:\n{OUTPUT_DIR}'
)

print(
    f'Model:\n{model_file}'
)

print(
    f'Metrics:\n'
    f'{OUTPUT_DIR / "metrics.json"}'
)

print(
    f'Predictions:\n'
    f'{prediction_file}'
)
