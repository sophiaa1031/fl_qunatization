import torch
import numpy as np
import copy
import random
import math


def quant_process(initial_dict,quantization_bits,device):

    print('quntized to {} bits'.format(quantization_bits))

    if quantization_bits == 32:
        return initial_dict

    if quantization_bits == 16: # 只是针对16bit的
        quantized_state_dict1 = copy.deepcopy(initial_dict)
        for key, value in initial_dict.items():
            quantized_state_dict1[key] = value.to(torch.float16).to(device)
        return quantized_state_dict1

    quantized_state_dict = copy.deepcopy(initial_dict)
    # 遍历字典中的所有值，找到最小值
    max_val = 0.0
    min_val = float('inf')
    for key, value in quantized_state_dict.items():
        min_val_in_tensor = value.abs().min().item()  # 获取张量中的最小值
        max_val_in_tensor = value.abs().max().item()  # 获取张量中的最大值
        if min_val_in_tensor < min_val:
            min_val = min_val_in_tensor
        if max_val_in_tensor > max_val:
            max_val = max_val_in_tensor
    print('the difference of min and max:{}'.format(max_val - min_val))
    for key, tensor in initial_dict.items():
        # min_val = tensor.min()
        # max_val = tensor.max()
        levels = 2 ** quantization_bits
        intervals = torch.linspace(min_val-(1e-10), max_val+(1e-6), steps=levels).to(device)
        indices = torch.bucketize(tensor.abs().to(device), intervals)-1  # index要减1从0开始
        indices[indices == levels] = levels - 1
        lower_bounds = intervals[indices]
        upper_bounds = intervals[indices + 1]
        probs = (upper_bounds - tensor.abs()) / (upper_bounds - lower_bounds)
        new_values = torch.where(torch.rand(tensor.shape).to(device)< probs, lower_bounds, upper_bounds)
        new_values = torch.where(tensor.to(device) < 0, -new_values, new_values)
        quantized_state_dict[key] = new_values.to(device)
    return quantized_state_dict
