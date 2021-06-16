#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/16 11:33
# @Author : zhanggong
# @Email : 601806353@qq.com
# @File : utils.py
# @Software: PyCharm
from losses import loss_func
import torch
import cv2

def collate_fn(batch):
    return tuple(zip(*batch))


def input_process(batch):
    batch_size = len(batch[0])##batch[0],image batch[1] = target
    inputs_batch = torch.zeros(batch_size,3,448,448)
    for i in range(batch_size):
        inputs_tmp = batch[0][i]
        inputs_tmp1 = cv2.resize(inputs_tmp.permute([1,2,0]).numpy(),(448,448)) #chw-->w,h,c
        inputs_tmp2 = torch.tensor(inputs_tmp1).permute([2, 0, 1])
        inputs_batch[i:i+1,...] = torch.unsqueeze(inputs_tmp2,0)#add axis
    return inputs_batch
def target_process(batch,gride_size = 7):
    pass





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
