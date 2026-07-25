import torch            # tensor 연산 
import torch.nn as nn   # 신경망 layer, loss 
from torch.utils.data import TensorDataset, DataLoader 

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")
# model = model.to(device)

# pytorch 모델은 nn.Module을 상속받아 만든다. 
class SimpleModel(nn.Module):
    # 모든 layer는 __init__ 에서 정의한다 
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,1)

    # 데이터가 흐르는 과정은 forward() 에서 작성 
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x 

model = SimpleModel(input_dim=10)

print(model)

'''
Dataset은 하나의 데이터가 어떻게 구성되는지를 나타내고, 
DataLoader는 Dataset을 batch단위로 가져온다. 
'''
X = torch.randn(1000,10)
y = X.sum(dim=1, keepdim=True)

dataset = TensorDataset(X,y)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

x_batch, y_batch = next(iter(train_loader))

print(x_batch.shape)    # torch.Size([32, 10])
print(y_batch.shape)    # torch.Size([32, 1])

"----------------------------------------------------------------------------"

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# 1. Device 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 2. 데이터 생성
torch.manual_seed(42)

X = torch.randn(1000, 10)
y = X.sum(dim=1, keepdim=True)

dataset = TensorDataset(X, y)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# 3. 모델 정의
class SimpleModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# 4. 모델 생성
model = SimpleModel(input_dim=10).to(device)


# 5. Loss와 optimizer
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 6. 학습
num_epochs = 20

for epoch in range(num_epochs):

    model.train()

    epoch_loss = 0.0

    for x_batch, y_batch in train_loader:

        # 데이터도 모델과 같은 device로 이동
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 이전 gradient 초기화
        optimizer.zero_grad()

        # Forward
        prediction = model(x_batch)

        # Loss 계산
        loss = loss_fn(prediction, y_batch)

        # Backward
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.6f}"
    )