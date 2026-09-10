
'''
Require python3.11 braindecode venv
'''

# %%
import argparse
import braindecode
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, EarlyStopping
from braindecode import EEGClassifier
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.datasets import create_from_mne_epochs
import torch
import torch.nn as nn
from braindecode.models import EEGNetv4
import numpy as np

from util.easy_imports import *

print(braindecode.__version__)

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
# 1. 定义EEGNet模型


def create_eegnet_rsvp(
    n_channels=64,          # EEG通道数
    n_classes=2,            # 二分类 (target vs non-target)
    input_time_length=1000,  # 时间采样点数 (取决于采样率)
    n_filters_time=8,       # 时间卷积滤波器数量
    n_filters_spat=16,      # 空间卷积滤波器数量
    filter_time_length=64,  # 时间卷积核长度
    pool_time_length=8,     # 时间池化长度
    pool_time_stride=4,     # 时间池化步长
    drop_prob=0.25,         # Dropout概率
    n_filters_2=16,         # 第二层卷积滤波器数量
    filter_time_length_2=8,  # 第二层时间卷积核长度
    pool_time_length_2=8,   # 第二层时间池化长度
    pool_time_stride_2=4,   # 第二层时间池化步长
    final_conv_length=2,    # 最终卷积长度
):
    """
    创建用于RSVP任务的EEGNet模型

    RSVP (Rapid Serial Visual Presentation) 特点:
    - 刺激呈现快速 (通常每秒10-20个刺激)
    - 需要快速检测目标刺激
    - 单次试验信号微弱，需要强大特征提取
    """

    model = EEGNetv4(
        n_channels=n_channels,
        n_classes=n_classes,
        input_window_samples=input_time_length,
        n_filters_time=n_filters_time,
        n_filters_spat=n_filters_spat,
        filter_time_length=filter_time_length,
        pool_time_length=pool_time_length,
        pool_time_stride=pool_time_stride,
        drop_prob=drop_prob,
        n_filters_2=n_filters_2,
        filter_time_length_2=filter_time_length_2,
        pool_time_length_2=pool_time_length_2,
        pool_time_stride_2=pool_time_stride_2,
        final_conv_length=final_conv_length,
    )

    # 转换为密集预测模型（对于分类任务很重要）
    # to_dense_prediction_model(model)

    return model


# %%
# 2. 更轻量级的RSVP专用EEGNet
class EEGNetRSVP(nn.Module):
    """为RSVP任务优化的轻量级EEGNet"""

    def __init__(
        self,
        n_channels=64,
        n_classes=2,
        sfreq=250,           # 采样率 (Hz)
        t_min=0.2,           # 分析时间窗起始 (秒)
        t_max=0.6,           # 分析时间窗结束 (秒)
    ):
        super().__init__()

        # 计算采样点
        n_samples = int((t_max - t_min) * sfreq)

        # Block 1: 时间卷积 + 空间卷积
        self.conv_block1 = nn.Sequential(
            # 时间卷积: (batch, 1, chans, time)
            nn.Conv2d(1, 16, (1, 64), padding='same'),
            nn.BatchNorm2d(16),
            # 空间卷积: 深度可分离卷积
            nn.Conv2d(16, 32, (n_channels, 1), groups=16),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 4)),
            nn.Dropout(0.25),
        )

        # Block 2: 可分离卷积
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding='same', groups=32),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 4)),
            nn.Dropout(0.25),
        )

        # 计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self.conv_block1(dummy)
            dummy = self.conv_block2(dummy)
            self.feature_dim = dummy.view(1, -1).shape[1]

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # 输入: (batch, channels, time)
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# %%
# 3. 使用Braindecode的完整训练流程
def prepare_rsvp_data(epochs_data, labels, sfreq):
    """
    准备RSVP数据用于训练

    Parameters:
    -----------
    epochs_data : numpy array, shape (n_epochs, n_channels, n_samples)
    labels : numpy array, shape (n_epochs,)
    sfreq : float, 采样率
    """
    from mne import EpochsArray, create_info

    # 创建MNE信息
    ch_names = [f'ch{i}' for i in range(epochs_data.shape[1])]
    ch_types = ['eeg'] * epochs_data.shape[1]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # 创建Epochs对象
    epochs = EpochsArray(epochs_data, info)
    epochs.annotations = labels  # 添加标签

    # 创建Braindecode数据集
    dataset = create_from_mne_epochs(epochs)

    return dataset


def create_rsvp_classifier(model, dataset, device='cuda'):
    """
    创建RSVP分类器
    """
    # 数据预处理
    # preprocessors = [
    #     Preprocessor('apply', fn=scale, factor=1e6),  # 转换为µV
    # ]

    # 准备训练数据
    train_set = dataset
    # 这里可以添加数据划分

    # 创建分类器
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        optimizer__weight_decay=0.001,
        batch_size=64,
        device=device,
        max_epochs=100,
        callbacks=[
            ('lr_scheduler', LRScheduler(policy='CosineAnnealingLR', T_max=100)),
            ('early_stop', EarlyStopping(patience=10)),
        ],
        train_split=predefined_split(train_set),  # 需要实际划分
    )

    return clf


# %%


# 4. 使用示例
def main():
    # How to prepare data
    if False:
        # 假设我们有RSVP数据
        # 典型RSVP参数:
        # - 采样率: 250 Hz
        # - 通道数: 64
        # - 目标时间窗: 0-0.6秒 (刺激后)

        # 模拟数据 (实际使用时替换为真实数据)
        n_samples = 150   # 0.6秒 * 250 Hz
        n_channels = 64
        n_epochs = 1000   # 1000个试次

        # 模拟数据
        X_sim = np.random.randn(n_epochs, n_channels,
                                n_samples).astype(np.float32)
        y_sim = np.random.randint(0, 2, size=n_epochs)

    # 创建模型
    model = create_eegnet_rsvp(
        n_channels=n_channels,
        n_classes=2,
        input_time_length=n_samples,
        drop_prob=0.25
    )

    # 或使用轻量级版本
    # model_light = EEGNetRSVP(
    #     n_channels=n_channels,
    #     n_classes=2,
    #     sfreq=250,
    #     t_min=0.0,
    #     t_max=0.6
    # )

    # 打印模型信息
    print("EEGNet for RSVP:")
    print(model)
    print(
        f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    dummy_input = torch.randn(4, n_channels, n_samples)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # 计算预测类别
    predictions = torch.argmax(output, dim=1)
    print(f"Predictions: {predictions}")

    return model


# %%
if __name__ == "__main__":
    main()


# %%
