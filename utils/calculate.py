import torch
import numpy as np
import copy
import random
import math


def subtract(params_a, params_b):
    w = copy.deepcopy(params_a)
    for k in w.keys():
            w[k] = w[k] - params_b[k]
    return w