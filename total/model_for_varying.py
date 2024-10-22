import torch
import torch.nn as nn

class VMFNet(nn.Module):
    def __init__(self):
        super(VMFNet,self).__init__()
        self.cov1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(1,5),bias=False,stride=(1,5))
        self.cov2 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(1,7),bias=False,stride=(1,7))
        self.cov3 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(1,9),bias=False,stride=(1,9))
        self.bn1 = nn.BatchNorm2d(8)
        self.ch_conv1 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(21,1),bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.ac1 = nn.ELU()
        self.dr1 = nn.Dropout(p=0.5)
        self.cov4_1 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(1,11),bias=False,stride=2,groups=16)
        self.cov4_2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(1,1),bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.ac2 = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1,4))
        self.dr2 = nn.Dropout(p=0.5)
        self.Flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128,out_features=5)
        self.reset_param()

    def reset_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = torch.reshape(x, (len(x), 1, 21, 170))
        time_x1 = self.cov1(x)
        time_x2 = self.cov2(x)
        time_x3 = self.cov3(x)
        time_x = torch.cat((time_x1,time_x2,time_x3),dim=3)
        time_x = self.bn1(time_x)
        time_x = self.dr1(time_x)
        x = self.ch_conv1(time_x)
        x = self.bn2(x)
        x = self.ac1(x)

        x = self.cov4_1(x)
        x = self.cov4_2(x)
        x = self.bn3(x)
        x = self.ac2(x)
        x = self.pool(x)
        x = self.dr2(x)
        x = self.Flatten(x)
        x = self.linear(x)
        return x

if __name__ =='__main__':
    model = VMFNet()
    input = model(torch.rand(size=(900,1,21,170)))
    print(input.shape)