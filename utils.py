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
from dataset import PennFudanDataset
from torch.utils.data import random_split
import numpy as np

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
def target_process(batch,grid_number = 7):
    batch_size=len(batch[0])
    target_batch = torch.zeros(batch_size, grid_number, grid_number, 5)
    labels = batch[1]
    for i in range(batch_size):
        batch_labels  = labels[i]
        number_box = len(batch_labels['boxes'])
        for wi in range(grid_number):
            for hi in range(grid_number):
                for bi in range(number_box):
                    bbox = batch_labels['boxes'][bi]
                    _,himg,wimg = batch[0][i].numpy().shape
                    bbox = bbox / torch.tensor([wimg, himg, wimg, himg])

                    center_x = (bbox[0] + bbox[2]) * 0.5
                    center_y = (bbox[1] + bbox[3]) * 0.5

                    if center_x<=(wi+1)/grid_number and center_x>=wi/grid_number and center_y<=(hi+1)/grid_number and center_y>= hi/grid_number:
                        cbbox = torch.cat([torch.ones(1), bbox])
                        # target_batch[i:i + 1, wi:wi + 1, hi:hi + 1, :] = torch.unsqueeze(cbbox, 0)
                        target_batch[i:i + 1, wi:wi + 1, hi:hi + 1,0] = cbbox[0]
                        target_batch[i:i + 1, wi:wi + 1, hi:hi + 1, 1] = cbbox[1]
                        target_batch[i:i + 1, wi:wi + 1, hi:hi + 1, 2] = cbbox[2]
                        target_batch[i:i + 1, wi:wi + 1, hi:hi + 1, 3] = cbbox[3]
                        target_batch[i:i + 1, wi:wi + 1, hi:hi + 1, 4] = cbbox[4]
    return target_batch




def train_one_epoch(model, optimizer, train_loader, device, epoch,scheduler):
    model.train()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    avg_loss = []
    for iter, batch in pbar:
        optimizer.zero_grad()
        # 取图片
        inputs = input_process(batch).to(device)
        # 取标注
        labels = target_process(batch).to(device)

        # 获取得到输出
        outputs = model(inputs)
        loss, lm, glm, clm = loss_func(outputs, labels)
        avg_loss.append(loss.data.item())
        loss.backward()
        optimizer.step()
        # print(torch.cat([outputs.detach().view(1,5),labels.view(1,5)],0).view(2,5))
        description = f"epoch:{epoch}    iter:{iter}    loss:{np.mean(avg_loss)}    lr:{optimizer.state_dict()['param_groups'][0]['lr']}"
        pbar.set_description(description)
    if scheduler is not None:
        scheduler.step()

def prepare_dataloader(cfg,transforms=None):
    dataset = PennFudanDataset(root=cfg.datapath,transforms=transforms)
    train_size = int(0.8*len(dataset))
    val_size = len(dataset)-train_size
    train_dataset,valid_dataset = random_split(dataset,[train_size,val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=cfg.n_job,collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size,
                                               shuffle=False, num_workers=cfg.n_job,collate_fn=collate_fn)
    return train_loader,valid_loader
