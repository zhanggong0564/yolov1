#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/18 11:20
# @Author : zhanggong
# @Email : 601806353@qq.com
# @File : config.py
# @Software: PyCharm
import torch


class Config(object):
    def __init__(self):
        self.num_classes  = 2
        self.batch_size  = 4
        self.epochs = 500
        self.lr = 1e-2
        self.momentum=0.9
        self.w_decay = 1e-5
        self.step_size =50
        self.gamma = 0.5
        self.datapath = r'H:\ubuntu18.04\paoyuan\maskrcnn\PennFudanPed'
        self.n_job = 4
        self.device = 'cuda'

