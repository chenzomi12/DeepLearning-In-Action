#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np


def max_pool_forward(x, pool_param):
    (N, C, H, W) = x.shape  # 获取输入矩阵的参大小
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = 1 + (H - pool_height) / stride  # 输出高
    W_prime = 1 + (W - pool_width) / stride  # 输出宽

    out = np.zeros((N, C, H_prime, W_prime))

    for n in xrange(N):
        for h in xrange(H_prime):
            for w in xrange(W_prime):
                # (h1, w1) 为当前pooling窗口第一个点
                # (h2, w2) 为当前pooling窗口最后一个点
                h1 = h * stride
                h2 = h * stride + pool_height
                w1 = w * stride
                w2 = w * stride + pool_width
                window = x[n, :, h1:h2, w1:w2]  # 当前pooling窗口
                win_l = window.reshape((C, pool_height * pool_width))
                out[n, :, h, w] = np.max(win_l, axis=1)
    return out


x = np.random.randint(5, size=(1, 1, 4, 4))
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
out = max_pool_forward(x, pool_param)
