#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/15 17:19
# @Author : zhanggong
# @Email : 601806353@qq.com
# @File : yolov1.py
# @Software: PyCharm
import torch.nn as nn
from model.backbone.vgg16 import VGG


class YOLOV1(nn.Module):
    def __init__(self,backbone,n_class=1):
       super(YOLOV1,self).__init__()
       cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
       if backbone=='vgg':
           self.backbone = VGG(cfg=cfg,use_bn=True)
       self.extractor = self.backbone.extractor
       self.avgpool = nn.AdaptiveAvgPool2d((7,7))
       # 决策层：检测层
       self.detector = nn.Sequential(
          nn.Linear(512*7*7,4096),
          nn.ReLU(True),
          nn.Dropout(),
          nn.Linear(4096,245),
       )
       self.n_class = n_class
       for m in self.modules():
           if isinstance(m,nn.Conv2d):
               nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias,0)
           elif isinstance(m,nn.BatchNorm2d):
               nn.init.constant_(m.weight,1)
               nn.init.constant_(m.bias,1)
           elif isinstance(m,nn.Linear):
               nn.init.normal_(m.weight,0,0.01)
               nn.init.constant_(m.bias,0)
    def forward(self,x):
        x = self.extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.detector(x)
        b,_ = x.shape
        x = x.view(b,7,7,4+self.n_class)
        return x