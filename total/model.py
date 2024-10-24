import math
from typing import Optional, List

import torch.nn as nn
import torch

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
RANDOM_STATE = SEED


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
        # self.reset_param()

    # def reset_param(self):
    #     for m in self.modules():
    #         if isinstance(m,nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight,mode='fan_in')
    #         elif isinstance(m,nn.Linear):
    #             nn.init.xavier_normal_(m.weight)
    #             nn.init.constant_(m.bias,0)

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


class binary_EEGNet(nn.Module):
    def __init__(self):
        super(binary_EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 100), padding=(0, 50), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(21, 1), bias=False, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5),
            # D*F1 == F2
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16), bias=False,
                                 padding=(0, 8), groups=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 100), padding=(0, 50), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(21, 1), bias=False, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5),
            # D*F1 == F2
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16), bias=False,
                                 padding=(0, 8), groups=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.attn = time_Attention_block(time_length=16 * (170 // 32) * 2,feed_mid_length=16 * (170 // 32))
        self.f1 = nn.Linear(16 * (170 // 32) * 2, 5)

    def forward(self, x1, x2):
        x1 = torch.reshape(x1, (len(x1), 1, 21, 170))
        x2 = torch.reshape(x2,(len(x2), 1, 21, 170))
        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x = torch.cat((x1, x2), dim=1).unsqueeze(dim=0)
        x = self.attn(x).squeeze(dim=0)
        return self.f1(x)


class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True,**kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self,query,key,value,mask: Optional[torch.Tensor] = None):
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)

        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)

        attn = self.dropout(attn)

        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)


class feed_forward(nn.Module):
    def __init__(self,input_length,mid_length):
        super(feed_forward,self).__init__()
        self.fc1 = nn.Linear(input_length,mid_length)
        self.fc2 = nn.Linear(mid_length,input_length)

    def forward(self,x):
        return self.fc2(self.fc1(x))

class time_Attention_block(nn.Module):
    def __init__(self,time_length,feed_mid_length,head_num=1,dropout_ratio=0.1,):
        super(time_Attention_block,self).__init__()
        self.attn = MultiHeadAttention(d_model=time_length,heads=head_num)
        self.feed_forward = feed_forward(input_length=time_length,mid_length=feed_mid_length)

    def forward(self,x):
        x = x + self.attn(x,x,x)
        x = x + self.feed_forward(x)
        return x


if __name__ =='__main__':
    model = binary_EEGNet()
    output = model(torch.rand((900,1,21,170)),torch.rand((900,1,21,170)))
    print(output.shape)
    # model = time_Attention_block(
    #     time_length=5,
    #     feed_mid_length=10
    # )
    # x = torch.rand((16,900,5))
    # output = model(x)
    # print(output.shape)