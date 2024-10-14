import torch
import torch.nn as nn
import torchsummary.torchsummary
class EEG_Deformer(nn.Module):
    #This deformer can deformer the time convolution, and the convolution can spread the whole EEG time segment

    def __init__(self,time_convolution_length=5,offset_length=5):
        super(EEG_Deformer,self).__init__()
        self.offset_conv = nn.Conv2d(in_channels=1,out_channels=5,kernel_size=(1,5),stride=1)


    def forward(self,x):
        return self.offset_conv(x)




    @staticmethod
    def get_position(convolution_length):
        a = range(-(convolution_length//2),(convolution_length+1)//2)

        return list(a)


if __name__ == '__main__':
    model = EEG_Deformer()
    output = model(torch.rand((900,1,21,170)))
    print(output.shape)