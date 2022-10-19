'''
RNN无法并行化
RNN、CNN无法考虑长距离的信息的相关性-在RNN上加上Attention，self-attention可以取代RNN

RNN模型注意力机制的改进
    一是先用一个注意力网络考虑其相关性，再送入RNN
    二是直接全部换成注意力网络，效率更高
CNN模型注意力机制的改进
    例如三个不同位置提取的特征（用不同卷积核），采用不同的权重，重新生成新的特征图谱，这个就叫注意力，
    再用常规的卷积核卷积生成特征图谱，最后将这两种融合起来（FM），常用融合技术：连接，点积

分类：
Soft attention：用所有的特征图谱，以不同的权重形成新的特征图谱。
Hard attention:
self attention：

Bottleneck Attention Module(BAM)
    Global avg pool->FC->FC + 1*1cov->3*3cov->3*3cov->1*1cov =>sigmoid(BAM层）

Convolutional Block Attention Module(CBAM)卷积块注意力模型
    。。。
    残差块+CBAM
    通道注意力+空间注意力

self attention:
    将每个单独的特征片段抽出来，给这些特征以处理（编码），直接让特征们互相匹配
    q（钥匙）:拿去给别人配对  k（锁）:跟别人的q配对 v（自己）：做最后的补充

应用：
用于取代RNN模型：transformer
用于增强CNN模块对长距离信息的分析能力：SAGAN


Spectral Normalization权重谱范数归一化：比WGAN-GP更好的技术
    思路：约束D和G的权值W，来满足1-L条件
    实现：pytorch中，调用utils.sn
SNGAN+SA->SAGAN

SA的具体实现：
    先是引入三个预处理，三个不同的卷积层，把每个Feature map处理成Key，Query，Value
    然后将每一个Feature map的Key和其他所有Feature map的Query相乘，然后乘以Value，得到所有的b，再将bi加权求和，得到b

SAGAN：
    将self-attention引入了SNGAN(WGAN的一种强化版)中

'''
import torch
import torch.nn as nn

#自注意力层
class Self_Attn(nn.Module):
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in=in_dim
        self.activation=activation
        self.with_attn=with_attn
        self.query_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))#可学习的参数

        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize,C,width,height=x.size()
        proj_query=self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)##B*(w*H)*C
        proj_key=self.key_conv(x).view(m_batchsize,-1,width*height)#B*C*(w*H)
        energy=torch.bmm(proj_query,proj_key)#B*(W*H)*(W*H)
        attention=self.softmax(energy)#B*(W*H)*(W*H)
        proj_value=self.value_conv(x).view(m_batchsize,-1,width*height)#B*C*(W*H)

        out=torch.bmm(proj_value,attention.permute(0,2,1))#B*(W*H)*C
        our=out.view(m_batchsize,C,width,height)#B*C*W*H

        out=self.gamma*out+x

        if self.with_attn:
            return out+x,attention
        else:
            return out+x

