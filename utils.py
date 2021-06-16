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
from tqdm import tqdm

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





def train_one_epoch(model, optimizer, train_loader, device, epoch,scheduler):
    model.train()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for iter, batch in pbar:
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
        description = f"epoch:{epoch}    iter:{iter}    loss:{loss.data.item()}    lr:{optimizer.state_dict()['param_groups'][0]['lr']}"
        pbar.set_description(description)
    if scheduler is not None:
        scheduler.step()
