import torch.nn as nn
import torch

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 100), padding=(0, 50), bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(21, 1), bias=False, groups=8)
        self.bn2 = nn.BatchNorm2d(16)
        self.ac2 = nn.ELU()
        self.av2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dr2 = nn.Dropout(p=0.5)
        # D*F1 == F2
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16), bias=False,
                                 padding=(0, 8), groups=16)
        self.conv3_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.ac3 = nn.ELU()
        self.av3 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dr3 = nn.Dropout(p=0.5)

        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(16*(170//32), 5)
        self.reset_param()

    def reset_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in')
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        x = torch.reshape(x, (len(x), 1, 21, 170))
        x = x[:, :, :, range(169)]#这个步骤可以去掉
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.av2(x)
        x = self.dr2(x)
        x = x[:, :, :, range(41)]#这个步骤可以去掉
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.ac3(x)
        x = self.av3(x)
        x = self.dr3(x)
        x = self.flatten(x)
        y = self.f1(x)
        return y