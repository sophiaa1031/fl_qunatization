#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w,w_global):
    w_avg = copy.deepcopy(w[0])
    w_global_saved = copy.deepcopy(w_global)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
        w_global_saved[k] += w_avg[k]
    return w_global_saved
