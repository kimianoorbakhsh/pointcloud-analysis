import torch
import torch.nn as nn
import torch.nn.functional as F



class Tnet(nn.Module):

    def __init__(self, k, device):
        super().__init__()

        self.k = k
        self.device = device

        self.mlp1 = nn.Conv1d(in_channels=k, out_channels=64, kernel_size=1)
        self.mlp2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.mlp3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.maxpool = nn.MaxPool1d(1024)

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.mlp2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.mlp3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = nn.MaxPool1d(x.shape[-1])(x)

        x = nn.Flatten()(x)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.fc3(x)

        init = torch.eye(self.k, requires_grad=True, device=self.device).repeat(batch_size, 1, 1)
        res = x.view((-1, self.k, self.k)) + init

        return res


class TransformNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.Tnet3 = Tnet(3, device=device)
        self.Tnet64 = Tnet(64, device=device)

        self.mlp1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)

        self.mlp2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)

        self.mlp3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        rot_mat3 = self.Tnet3(x)
        x = x.transpose(1, 2)

        x = torch.bmm(x, rot_mat3)

        x = x.transpose(1, 2)

        x = self.mlp1(x)
        x = self.bn1(x)
        x = self.relu(x)

        rot_mat64 = self.Tnet64(x)
        x = x.transpose(1, 2)

        x = torch.bmm(x, rot_mat64)

        x = x.transpose(1, 2)

        x = self.mlp2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.mlp3(x)
        x = self.bn3(x)

        x = nn.MaxPool1d(x.shape[-1])(x)

        x = nn.Flatten()(x)

        return x, rot_mat3, rot_mat64


class Pointnet(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.transformnet = TransformNet(device=device)

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=40)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        res, rot3, rot64 = self.transformnet(x)

        res = self.fc1(res)
        res = self.bn1(res)
        res = self.relu(res)

        res = F.dropout(res, p=0.3, training=self.training)

        res = self.fc2(res)
        res = self.bn2(res)
        res = self.relu(res)

        res = F.dropout(res, p=0.3, training=self.training)

        res = self.fc3(res)

        return res, rot3, rot64


##loss function with transformation regularizer

def loss_function(output, true, rot64, device, alpha = 0.001):
  criterion = nn.CrossEntropyLoss()

  batch_size = output.shape[0]

  i64 = torch.eye(64, requires_grad=True, device=device).repeat(batch_size, 1, 1)

  mat64 = torch.bmm(rot64, rot64.transpose(1, 2))

  dif64 = nn.MSELoss(reduction='sum')(mat64, i64) / batch_size

  loss1 = criterion(output, true)
  loss2 = dif64
  loss = loss1 + alpha * loss2

  return loss, loss1, loss2