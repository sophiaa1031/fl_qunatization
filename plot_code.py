import os
import matplotlib.pyplot as plt
import numpy as np

# 指定包含文件的目录路径
directory_path = "figures_data/cifar_loss_mlp30"
# "figures_data/mnist_loss_mlp10"
def loss_nozerotrust(path):
    # 获取目录下的所有文件
    files = os.listdir(directory_path)

    # 初始化一个列表来存储所有的曲线数据
    all_data = []

    # 循环读取每个文件的数据并存储到all_data列表中
    for file in files:
        file_path = os.path.join(directory_path, file)
        legend_label = os.path.splitext(file)[0]  # 使用文件名作为legend标签
        data = np.loadtxt(file_path)
        plt.plot(range(1, len(data)+1), data, label=legend_label)

    # 显示图形
    plt.xlabel("FL iterations", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)  # 显示legend标签
    plt.title(directory_path)
    # plt.savefig('save/le1_iid_loss.png')  # 保存为PNG格式图片
    plt.grid()
    plt.show()

loss_nozerotrust(directory_path)