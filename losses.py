#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/15 17:37
# @Author : zhanggong
# @Email : 601806353@qq.com
# @File : losses.py
# @Software: PyCharm

import torch


def  loss_func(outputs,labels):
    assert ( outputs.shape == labels.shape),"outputs shape[%s] not equal labels shape[%s]"%(outputs.shape,labels.shape)
    b,w,h,c = outputs.shape
    loss = 0

    conf_loss_matrix = torch.zeros(b, w, h)
    geo_loss_matrix = torch.zeros(b, w, h)
    loss_matrix = torch.zeros(b, w, h)


    for bi in range(b):
        for wi in range(w):
            for hi in range(h):
                detect_vector = outputs[bi, wi, hi]
                gt_dv = labels[bi, wi, hi]

                conf_pred = detect_vector[0]
                conf_gt = gt_dv[0]

                x_pred = detect_vector[1]
                x_gt = gt_dv[1]

                y_pred = detect_vector[2]
                y_gt = gt_dv[2]

                w_pred = detect_vector[3]
                w_gt = gt_dv[3]

                h_pred = detect_vector[4]
                h_gt = gt_dv[4]

                loss_confidence = (conf_pred - conf_gt) ** 2

                loss_geo = (x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2 + (w_pred - w_gt) ** 2 + (h_pred - h_gt) ** 2

                loss_geo = conf_gt * loss_geo

                loss_tmp = loss_confidence + 0.3 * loss_geo
                loss += loss_tmp

                conf_loss_matrix[bi, wi, hi] = loss_confidence
                geo_loss_matrix[bi, wi, hi] = loss_geo
                loss_matrix[bi, wi, hi] = loss_tmp
    print(geo_loss_matrix)
    print(outputs[0, :, :, 0] > 0.5)
    return loss, loss_matrix, geo_loss_matrix, conf_loss_matrix




    