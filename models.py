import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""Classic CNN model"""
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64*28*28, 512)
        self.fc2 = nn.Linear(512, 10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(-1, 64*28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = F.relu(x)
        return x

"""Residual CNN model"""
class MyResidualNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10):
        super(MyResidualNet, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloques residuales reducidos
        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.res_block2 = ResidualBlock(64, 128, stride=2)

        # Capa completamente conectada
        self.fc1 = None  # Se inicializará dinámicamente
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.res_block1(x)
        x = self.maxpool(x)

        x = self.res_block2(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)  # Aplanar el tensor

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 256).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

"""Transfer learning model"""
class MyCNNWithTransferLearning(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNNWithTransferLearning, self).__init__()

        # Cargar el modelo preentrenado DenseNet121
        self.densenet = models.densenet121(pretrained=True)

        # Congelar las capas del modelo preentrenado para evitar sobreajuste
        for param in self.densenet.features.parameters():
            param.requires_grad = False

        # Modificar la capa final del DenseNet para ajustarla a las clases deseadas
        self.densenet.classifier = nn.Sequential(
            nn.Linear(self.densenet.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Forward pass a través de DenseNet
        return self.densenet(x)