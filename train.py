#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/16 11:33
# @Author : zhanggong
# @Email : 601806353@qq.com
# @File : train.py
# @Software: PyCharm
from losses import loss_func
import torch


def input_process(batch):
    batch_size = len(batch[0])
    input_batch = torch.zeros(batch_size,3,448,448)
    for i in range(batch_size):
        inputs_tmp = batch[0][i]





def train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq,scheduler):
    model.train()
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # 取图片
        inputs = input_process(batch)
        # 取标注
        labels = target_process(batch)

        # 获取得到输出
        outputs = model(inputs)
        loss, lm, glm, clm = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(torch.cat([outputs.detach().view(1,5),labels.view(1,5)],0).view(2,5))
        if iter % 10 == 0:
            #    print(torch.cat([outputs.detach().view(1,5),labels.view(1,5)],0).view(2,5))
            print("epoch{}, iter{}, loss: {}, lr: {}".format(epoch, iter, loss.data.item(),
                                                             optimizer.state_dict()['param_groups'][0]['lr']))

        # print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        # print("*"*30)
        # val(epoch)
    scheduler.step()
