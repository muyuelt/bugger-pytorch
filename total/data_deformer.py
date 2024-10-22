import torch
import torch.nn as nn
import torchsummary.torchsummary
class EEG_Deformer(nn.Module):
    #This deformer can deformer the time convolution, and the convolution can spread the whole EEG time segment

    def __init__(self,time_convolution_length=5,offset_length=5):
        super(EEG_Deformer,self).__init__()
        self.time_convolution_length = time_convolution_length
        self.offset_length = offset_length
        self.offset_conv = nn.Conv2d(in_channels=1,out_channels=time_convolution_length,kernel_size=(1,time_convolution_length),stride=1)
        self.tanh = nn.Tanh()


    def forward(self,x):
        x_padding = self.padding_zero(x)
        offset_step = self.tanh(self.offset_conv(x_padding))
        offset_num = self.calculate_offset_num(x,offset_step)
        return offset_num

    def padding_zero(self,x:torch.Tensor):
        b,f,c,t = x.shape
        p1 = torch.zeros((b,f,c,self.time_convolution_length//2))
        p2 = torch.zeros((b,f,c,(self.time_convolution_length-1)//2))
        return torch.cat(tensors=(p1,x,p2),dim=3)

    def calculate_offset_num(self,x:torch.Tensor,offset_step:torch.Tensor):
        offset_tensor = offset_step.permute(0,2,3,1)
        offset_position = self.get_convolution_position(offset_tensor)
        #获取带头尾信息的EEG数据,用于移动数据
        x_add_tail_head = self.padding_tail_and_head(x)
        #适应填充后的x向量
        offset_position = offset_position + self.time_convolution_length//2+self.offset_length
        #计算具体x内容
        offset_left = offset_position.detach().floor()
        offset_right = offset_left+1
        offset_left_ratio = offset_position - offset_left
        offset_right_ratio = offset_right - offset_position
        offset_left_num_0=[]
        for batch_size_i in range(len(offset_left)):
            batch_size = offset_left[batch_size_i]
            offset_left_num_1 =[]
            for channel_size_i in range(len(batch_size)):
                channel_size = batch_size[channel_size_i]
                offset_left_num_2 =[]
                for time_size_i in range(len(channel_size)):
                    time_size = channel_size[time_size_i]
                    offset_left_num_3 =[]
                    for idx in time_size:
                        offset_left_num_3.append(x_add_tail_head[batch_size_i,0,channel_size_i,int(idx)].item())
                    offset_left_num_2.append(offset_left_num_3)
                offset_left_num_1.append(offset_left_num_2)
            offset_left_num_0.append(offset_left_num_1)
        offset_left_num = torch.tensor(offset_left_num_0)
        offset_right_num_0=[]
        for batch_size_i in range(len(offset_right)):
            batch_size = offset_right[batch_size_i]
            offset_right_num_1 =[]
            for channel_size_i in range(len(batch_size)):
                channel_size = batch_size[channel_size_i]
                offset_right_num_2 =[]
                for time_size_i in range(len(channel_size)):
                    time_size = channel_size[time_size_i]
                    offset_right_num_3 =[]
                    for idx in time_size:
                        offset_right_num_3.append(x_add_tail_head[batch_size_i,0,channel_size_i,int(idx)].item())
                    offset_right_num_2.append(offset_right_num_3)
                offset_right_num_1.append(offset_right_num_2)
            offset_right_num_0.append(offset_right_num_1)
        offset_right_num = torch.tensor(offset_left_num_0)
        offset_num = torch.mul(offset_left_ratio,offset_left_num)+torch.mul(offset_right_ratio,offset_right_num)
        offset_num = torch.sum(offset_num,dim=3).unsqueeze(dim=3).permute(0,3,1,2)
        return offset_num




    def get_convolution_position(self,offset_tensor):
        #设置每个点的偏移访问位于（-offset_length,offset_length)之间
        offset_tensor = offset_tensor * self.offset_length
        #(1,1,1,convolution_length)
        offset_position_n = torch.tensor(self.get_position(self.time_convolution_length)).reshape(1,1,1,-1)
        #(1,1,time,1)
        offset_position_0 = torch.arange(0,offset_tensor.shape[2],1).reshape(1,1,-1,1)
        #为offset添加中心点信息
        offset_tensor = offset_tensor + offset_position_0
        #为offset添加每个偏移量对应的点信息
        offset_tensor = offset_tensor + offset_position_n

        return offset_tensor

    def padding_tail_and_head(self,x:torch.tensor):
        x_time_length = x.shape[3]
        x_head = x[:,:,:,:self.time_convolution_length//2+self.offset_length]
        x_tail = x[:,:,:,x_time_length-((self.time_convolution_length-1)//2+self.offset_length):x_time_length+1]
        x_new = torch.cat((x_tail,x,x_head),dim=3)
        return x_new

    def calculate_x_offset(self,x,offset):
        pass

    @staticmethod
    def get_position(convolution_length):
        a = range(-(convolution_length//2),(convolution_length+1)//2)
        return list(a)


if __name__ == '__main__':
    model = EEG_Deformer(7,2)
    output = model(torch.randn((10,1,21,170))*5)
    print(output[0,:,:,:])
    # offset_p = model.get_convolution_position(torch.randn((900,21,170,7)))
    # print(offset_p.shape)
    # model.calculate_offset_num(torch.randn((900,1,21,170)),output)