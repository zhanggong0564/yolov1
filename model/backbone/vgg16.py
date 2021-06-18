#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/15 17:18
# @Author : zhanggong
# @Email : 601806353@qq.com
# @File : vgg16.py
# @Software: PyCharm
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,cfg,in_channels=3,use_bn =False):
        super(VGG, self).__init__()
        layers = []
        for v in cfg:
            if v=='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1,bias=not use_bn)
                if use_bn:
                    layers+=[conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                else:
                    layers+=[conv2d,nn.ReLU(inplace=True)]
                in_channels = v

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = self.features(x)
        x_fea = x
        x = self.avgpool(x)
        x_avg =x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, x_fea, x_avg
    def extractor(self,x):
        x = self.features(x)
        return x

if __name__ == '__main__':
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']