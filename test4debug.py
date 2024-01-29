import torch
import torch.nn as nn


## test1
def test1():
    value = torch.tensor(3.141592653589793, dtype=torch.float32)
    value_float16 = value.to(torch.float16)
    formatted_value = "{:.15f}".format(value)
    formatted_value_float16 = "{:.15f}".format(value_float16)
    print(formatted_value, value_float16, formatted_value_float16)


## test2
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_conv_and_fc_layers(model):
    num_conv_layers = 0
    num_fc_layers = 0

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            num_conv_layers += 1
        elif isinstance(layer, nn.Linear):
            num_fc_layers += 1

    return num_conv_layers, num_fc_layers


# 创建ResNet模型

# 创建ResNet模型
# net = ResNet(BasicBlock, [3, 3, 3])
#
# # 计算模型参数数量
# num_conv_layers, num_fc_layers = count_conv_and_fc_layers(net)
# print(f"Total number of parameters in the ResNet model: {num_conv_layers, num_fc_layers}")

tensor = torch.tensor([-1, 1.5, -2, 2.5, -3])
lower_bounds = torch.tensor([1, 1, 1, 1, 1])
upper_bounds = torch.tensor([3, 3, 3, 3, 3])

probs = (upper_bounds - tensor.abs()) / (upper_bounds - lower_bounds)
new_values = torch.where(torch.rand(tensor.shape) < probs, lower_bounds, upper_bounds)
new_values = torch.where(tensor < 0, -new_values, new_values)
print(new_values)
