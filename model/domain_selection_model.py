import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import pickle
import numpy as np 
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
######## plot ########
def plot(y):
    x = [i for i in range(len(y))]
    plt.plot(x, y)
    plt.xlabel("time")
    plt.ylabel("voltage")
    plt.show()

######## 데이터 처리 ########
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "HUST"

pkl_file = DEFAULT_DATA_DIR / "HUST_1-1.pkl"

with open(pkl_file, "rb") as f:
    data = pickle.load(f)

cycle_data = data["cycle_data"]

cycle_data_0 = cycle_data[0]

voltage = np.array(cycle_data_0['voltage_in_V'])

print(voltage.shape)
# plot(voltage)

# 256 크기로 보간 
def resample_to_fixed_length(
    signal: np.ndarray,
    target_length: int = 256,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)

    # 기존 신호의 상대적 위치
    original_position = np.linspace(
        0.0,
        1.0,
        num=len(signal),
    )

    # 새롭게 만들 256개 위치
    target_position = np.linspace(
        0.0,
        1.0,
        num=target_length,
    )

    # 선형 보간
    resampled_signal = np.interp(
        target_position,
        original_position,
        signal,
    )

    return resampled_signal.astype(np.float32)

voltage = resample_to_fixed_length(voltage)
print(voltage.shape)
# plot(voltage)

# min-max 정규화
def min_max_normalize(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    min_value = float(x.min())
    max_value = float(x.max())

    normalized = (x - min_value) / (max_value - min_value)

    return normalized

voltage = min_max_normalize(voltage)
# print(voltage)
# plot(voltage)

######## 모델 정의 ######## 

class Domain_invarient_feature_generator(nn.Module):
    def __init__(self,input_channels,*args, **kwargs):
        super().__init__(*args, **kwargs)
        ##### cnn #####
        cnn_layer = [
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=33, stride=1, padding=16),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        ]
        for _ in range(5):
            cnn_layer.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=33, stride=1, padding=16))
            cnn_layer.append(nn.BatchNorm1d(num_features=64))
            cnn_layer.append(nn.LeakyReLU())
            cnn_layer.append(nn.MaxPool1d(kernel_size=2, stride=2))
        cnn_layer.append(nn.Flatten())
        cnn_layer.append(nn.Linear(in_features=64*4,out_features=64))
        self.cnn = nn.Sequential(*cnn_layer)

        ##### gru ##### 
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.gru_dense = nn.Linear(in_features=256, out_features=64)

    def forward(self, x):
        # ======================================
        # CNN stream
        # ======================================
        # CNN은 [B, C, L] 형태 사용
        cnn_feature = self.cnn(x)

        # ======================================
        # GRU stream
        # ======================================
        # GRU는 batch_first=True일 때 [B, L, C]
        gru_input = x.permute(0, 2, 1)

        # gru_output: [B, 256, 256]
        # hidden:     [1, B, 256]
        gru_output, hidden = self.gru(gru_input)

        # 마지막 GRU layer의 hidden state
        # [1, B, 256] → [B, 256]
        last_hidden = hidden[-1]

        # [B, 256] → [B, 64]
        gru_feature = self.gru_dense(last_hidden)

        ## concat
        cnn_gru_feature = torch.concat([cnn_feature,gru_feature],dim=1)

        return cnn_gru_feature

class Soh_predictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last_dense = nn.Linear(128, 1)
    def forward(self, x):
        predicted_soh = self.last_dense(x)
        return predicted_soh

class Full_model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.difg = Domain_invarient_feature_generator(input_channels=256)
        self.soh_predictor = Soh_predictor()
    def forward():
        pass
    

model = Domain_invarient_feature_generator(input_channels=256)
print(model)
        
