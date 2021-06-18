#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/16 13:54
# @Author : zhanggong
# @Email : 601806353@qq.com
# @File : train.py
# @Software: PyCharm
from config import Config
from utils import *
from model.yolov1 import YOLOV1
import torch
from dataset import get_transform
cfg = Config()
def main():
    train_loader,valid_loader = prepare_dataloader(cfg,get_transform(train=False))
    model = YOLOV1('vgg').to(cfg.device)
    # optimizer = torch.optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum,weight_decay=cfg.w_decay)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    for epoch in range(cfg.epochs):
        train_one_epoch(model,optimizer,train_loader,device=cfg.device,epoch=epoch,scheduler=scheduler)


if __name__ == '__main__':
    main()