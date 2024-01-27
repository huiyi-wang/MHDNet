import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import math

class HKA(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        return u * attn


class Attention1(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()

        self.proj_1 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = HKA(d_model)
        self.proj_2 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class HKA2(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv_spatial(x)
        attn = self.conv1(attn)
        return u * attn


class Attention2(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = HKA2(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Hierarchical_Attention(nn.Module):
    def __init__(self,classes=1000,dim_in=768,reduce_dim=256,Fs_size = 384,gamma=0.1):
        super().__init__()
        self.classes = classes
        self.reduce_dim = reduce_dim
        self.gamma = gamma

        self.layer = nn.Linear(Fs_size, 1)
        init.normal_(self.layer.weight, std=0.01)     # 初始化权重
        init.constant_(self.layer.bias, 0)            # 初始化偏执

        # Fq'
        self.layer_q1 = nn.Linear(in_features=dim_in, out_features=self.reduce_dim)      # fc1
        init.normal_(self.layer_q1.weight, std=0.01)
        init.constant_(self.layer_q1.bias, 0)

        self.layer_q2 = nn.Linear(in_features=dim_in, out_features=self.reduce_dim)      # fc2
        init.normal_(self.layer_q2.weight, std=0.01)
        init.constant_(self.layer_q2.bias, 0)

        self.layer_q3 = nn.Linear(in_features=dim_in, out_features=self.reduce_dim)      # fc3
        init.normal_(self.layer_q3.weight, std=0.01)
        init.constant_(self.layer_q3.bias, 0)

        # Fs'
        self.layer_k1 = nn.Linear(in_features=dim_in, out_features=self.reduce_dim)      # fc4
        init.normal_(self.layer_k1.weight, std=0.01)
        init.constant_(self.layer_k1.bias, 0)

        self.layer_k2 = nn.Linear(in_features=dim_in, out_features=self.reduce_dim)      # fc5
        init.normal_(self.layer_k2.weight, std=0.01)
        init.constant_(self.layer_k2.bias, 0)

        self.layer_k3 = nn.Linear(in_features=dim_in, out_features=self.reduce_dim)      # fc6
        init.normal_(self.layer_k3.weight, std=0.01)
        init.constant_(self.layer_k3.bias, 0)

        self.attention1 = Attention1()
        self.attention2 = Attention2()

        self.output_layer = nn.Linear(in_features=dim_in*2,out_features=dim_in)
        init.normal_(self.output_layer.weight, std=0.01)
        init.constant_(self.output_layer.bias, 0)

    def forward(self,query_feature,Input_feature,):
        # query_feature:Transformer_Block的输出            [64,192 ,768]
        # Input_feature:CNN空间先验模块的输出                [64,1008,768]

        # query_feature = self.attention1(query_feature)  # 注意attention是卷积操作1
        # query_feature = self.attention2(query_feature)  # 注意attention也是卷积操作2
        # Input_feature = self.attention1(Input_feature)
        # Input_feature = self.attention2(Input_feature)

        q1_matrix = self.layer_q1(query_feature)                # [64,192,256]
        q1_matrix = q1_matrix - q1_matrix.mean(1, keepdim=True)

        q2_matrix = self.layer_q2(query_feature)                # [64,192,256]
        q2_matrix = q2_matrix - q2_matrix.mean(1, keepdim=True)

        q3_matrix = self.layer_q3(query_feature)                # [64,192,256]
        q3_matrix = q3_matrix - q3_matrix.mean(1, keepdim=True)

        k1_matrix = self.layer_k1(Input_feature)                # [64,1008,256]
        k1_matrix = k1_matrix - k1_matrix.mean(1, keepdim=True)

        k2_matrix = self.layer_k2(Input_feature)                # [64,1008,256]
        k2_matrix = k2_matrix - k2_matrix.mean(1, keepdim=True)

        k3_matrix = self.layer_k3(Input_feature)                # [64,1008,256]
        k3_matrix = k3_matrix - k3_matrix.mean(1, keepdim=True)

        attention_weight1 = torch.bmm(q1_matrix, q2_matrix.transpose(1, 2)) / math.sqrt(self.reduce_dim)    # [64,192,192]
        attention_weight1 = F.softmax(attention_weight1, dim=2)

        attention_weight2 = torch.bmm(k1_matrix, k2_matrix.transpose(1, 2)) / math.sqrt(self.reduce_dim)    # [64,1008,1008]
        attention_weight2 = F.softmax(attention_weight2, dim=2)

        attention_weight3 = torch.bmm(q3_matrix, k3_matrix.transpose(1, 2)) / math.sqrt(self.reduce_dim)    # [64,192,1008]
        attention_weight3 = F.softmax(attention_weight3, dim=2)

        attention_feature1 = torch.bmm(attention_weight1, query_feature)      # [64,192,768]

        hidden_feature = self.layer(attention_weight2)                        # [64,1008,1]
        hidden_feature = F.softmax(hidden_feature, dim=1)

        add = attention_weight3 + hidden_feature.transpose(1, 2)              # [64,192,1008]
        attention_feature2 = torch.bmm(add,Input_feature)                     # [64,192,768]
        adaptive_attention_feature1 = torch.cat([attention_feature1, attention_feature2], dim=2)  # [64,193,1536]  (1526 = 768 * 2)
        adaptive_attention_feature2 = self.output_layer(adaptive_attention_feature1)
        return adaptive_attention_feature1,adaptive_attention_feature2


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    xz = torch.randn([64,3072,768])
    x1 = torch.randn([64, 768,768])
    x2 = torch.randn([64, 192,768])
    x3 = torch.randn([64, 48, 768])
    cc = tuple[x1,x2,x3]
    c = torch.cat([x1,x2,x3],dim=1)           # [64,1008,768]
    x = torch.randn([64,192,768])           # [64,192 ,768]
    # x = x.to(device)
    X = torch.randn([64, 1024, 14, 14])
    model1 = Hierarchical_Attention(classes=1000,dim_in=768,reduce_dim=256,Fs_size = 1008,gamma=0.1)
    # model.to(device)
    y1 = model1(query_feature=x,Input_feature=c)
    print('y.shape:',y1[0].shape)
    print('y.shape:',y1[1].shape)

    # 下面开始进行attention模块的测试
    xx1 = torch.randn([64,64,96,32])
    xx2 = torch.randn([64,128,48,16])
    xx3 = torch.randn([64,256,24,8])
    model2 = Attention1(d_model=64)
    model3 = Attention1(d_model=128)
    model4 = Attention1(d_model=256)
    Y1 = model2(xx1)        # [64, 64, 96, 32]
    Y2 = model3(xx2)        # [64, 128, 48, 16]
    Y3 = model4(xx3)        # [64, 256, 24, 8]
    print(Y1.shape)
    print(Y2.shape)
    print(Y3.shape)

    # 下面开始进行Hierarchical_Attention_separately模块测试
    Model = Hierarchical_Attention_separately()
    YY = Model(query_feature=x,Input_feature1=x1,Input_feature2=x2,Input_feature3=x3)
    print('YY.shape',YY.shape)