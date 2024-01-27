import torch
import torch.nn as nn
from .resnet import ResNet, Bottleneck
from .ViT import vit_base_patch16_224
from .FCU import FCUUp, FCUDown
from .SFF import Injector
from functools import partial
from .baseline import Baseline


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

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


class Resnet50(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Resnet50, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x



class ConformReid(nn.Module):
    def __init__(self, num_classes, cfg):
        super(ConformReid, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        self.spm = Baseline(last_stride=1, model_path="./pretrain_model/resnet50.pth", model_name='resnet50', pretrain_choice='imagenet')

        self.base = vit_base_patch16_224(img_size=cfg.INPUT.SIZE_TRAIN,stride_size=cfg.MODEL.STRIDE_SIZE,
                                         drop_path_rate=cfg.MODEL.DROP_PATH,drop_rate= cfg.MODEL.DROP_OUT,
                                         attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]
        # 1 stage
        self.conv_1 = ConvBlock(inplanes=64, outplanes=256, res_conv=True, stride=1)
        self.cnn_block = ConvBlock(inplanes=256, outplanes=256, res_conv=False, stride=1, groups=1)
        self.fusion_block = ConvBlock(inplanes=256, outplanes=256, groups=1)

        self.squeeze_block = FCUDown(inplanes=256 // 4, outplanes=768, dw_stride=4)  # CNN feature maps -> Transformer patch embeddings
        self.expand_block  = FCUUp(inplanes=768, outplanes=256 // 4, up_stride=4)    # Transformer patch embeddings -> CNN feature maps
        self.injector = Injector(dim=768, reduce_dim=256, fs_size1=384, gamma=0.1, init_values=1.0,with_cp=False, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.dw_stride = 4

        norm_layer = nn.LayerNorm
        self.norm = norm_layer(768)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


        self.classifier_CNN = nn.Linear(2048, self.num_classes, bias=False)
        self.classifier_CNN.apply(weights_init_classifier)

        self.bottleneck_CNN = nn.BatchNorm1d(2048)
        self.bottleneck_CNN.bias.requires_grad_(False)
        self.bottleneck_CNN.apply(weights_init_kaiming)

    def forward(self, x):

        global_feat_transformer,outputs = self.base(x)
        #
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))        # [64,64,64,32]
        x = self.conv_1(x_base, return_x_2=False)                        # [64,256,64,32]

        #
        for i in range(len(outputs)):
            y = outputs[i]
            x,x2 = self.cnn_block(x)
            _, _, H, W = x2.shape
            y = self.expand_block(y,H//self.dw_stride,W//self.dw_stride)        # y [64,129,768] -> [64,128,768] -> [64,768,16,8] -> [64,64,16,8] -> [64,64,64,32]
            x,x2 = self.fusion_block(x, y, return_x_2=True)
            if i == len(outputs)-1:
                x_st = self.squeeze_block(x2,outputs[i])

        x_st = x_st + outputs[11]
        x_st = self.norm(x_st)
        x_st = x_st[:, 0]

        feat = self.bottleneck(x_st)
        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, x_st
        else:
            if self.neck_feat == 'after':
                return feat

            else:
                return x_st


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))



def make_model(cfg, num_class):
    if cfg.MODEL.NAME == 'transformer':
        model = ConformReid(num_class, cfg)
        print('===========backbone is transformer===========')
    else:
        model = Resnet50(num_class, cfg)
        print('===========backbone is ResNet===========')
    return model
