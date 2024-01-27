# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .resnet import ResNet, BasicBlock, Bottleneck
from .senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .resnet_ibn_a import resnet50_ibn_a

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048
    # in_planes = 1024
    # in_planes = 512
    # in_planes = 256
    def __init__(self, last_stride, model_path, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)


        self.fc1 = nn.Conv2d(in_channels=    256, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(in_channels=2 * 256, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(in_channels=4 * 256, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(in_channels=8 * 256, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        # x [64,3,384,128]
        x_3,x_4 = self.base(x)           # x_4 [64,2048,24,8]
        # xx1 = self.fc1(x_1)                   # [64,768,96,32]
        # xx2 = self.fc2(x_2)                   # [64,768,48,16]
        xx3 = self.fc3(x_3)                   # [64,768,24,8]
        xx4 = self.fc4(x_4)                   # [64,768,24,8]

        bs, dim, _, _ = xx3.shape
        # xx1 = xx1.view(bs, dim, -1).transpose(1, 2)     # [64,768,96,32] -> [64, 3072,768]
        # xx2 = xx2.view(bs, dim, -1).transpose(1, 2)     # [64,768,48,16] -> [64, 768, 768]
        xx3 = xx3.view(bs, dim, -1).transpose(1, 2)     # [64,768,24,8]  -> [64, 192, 768]
        xx4 = xx4.view(bs, dim, -1).transpose(1, 2)     # [64,768,24,8]  -> [64, 192, 768]
        c = torch.cat([xx3,xx4],dim=1)  # [64, 384, 768]

        global_feat = self.gap(x_4)                    # global_feat.shape (64, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  #  (64, 2048)

        return c,global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


class Spatial_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = Baseline()

    def forward(self,x):
        x = self.base(x)
        pass
