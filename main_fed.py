#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,  cifar_noniid, cifar100_iid
from utils.options import args_parser
from utils.calculate import subtract
from utils.quant_process import quant_process
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResNet, BasicBlock
from models.Fed import FedAvg
from models.test import test_img
import os
import sys
import time

def saveData(my_list, file_path, way):
    try:
        # 将列表元素转换为字符串并以换行符分隔
        if way == 'w':
            list_as_str = '\n'.join(map(str, my_list))
        else:
            list_as_str = '\t'.join(map(str, my_list))+'\n'

        # 打开文件并写入列表内容
        with open(file_path, way) as file:
            file.write(list_as_str)

        print(f"列表已成功保存到文件: {file_path}")
    except Exception as e:
        print(f"保存列表到文件时发生错误: {str(e)}")

def mainFunction(bit):
    args.quantization_bits = bit
    print('args.quantization_bits', args.quantization_bits)
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=30, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet':
        net_glob = ResNet(BasicBlock, [3, 3, 3],args.num_classes, args.num_channels).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc_saved = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            start_time = time.time()
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            end_time = time.time()
            training_time = end_time - start_time
            print(f"模型训练时间：{training_time:.2f} 秒")
            w_update_local = subtract(w, w_glob)
            local_quant = quant_process(w_update_local, args.quantization_bits,args.device)
            # w_update_local, communication_cost, mse_error = local_quant.quant()
            # if iter == 0 and idx == 0:
                # print("The quantizationnbit is: {}, and the size is: {}".format(w['layer_input.weight'].dtype,sys.getsizeof(w)*8))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(local_quant)
            else:
                w_locals.append(copy.deepcopy(local_quant))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals, w_glob)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # testing
        net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)

        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # print("Training accuracy: {:.2f}, loss : {:.2f}".format(acc_train,loss_train))
        print("Testing accuracy: {:.2f}, loss : {:.2f}".format(acc_test,loss_test))
        acc_saved.append(acc_test.item())

    # save data
    subfolder = 'debug' # local_debug, debug
    folder_path = 'figures_data/'+subfolder+'/'
    # loss 保存
    file_path = folder_path+str(bit)+'_bit.txt'  # 指定文件路径(remove_malicious, attack, all_benign,attack_middle)
    saveData(loss_train, file_path,'w')
    # acc 保存
    file_path_acc = folder_path+str(bit)+'_bit_acc.txt'  # 指定文件路径(remove_malicious, attack, all_benign,attack_middle)
    saveData(acc_saved, file_path_acc, 'w')

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # args.dataset = 'cifar100'
    # args.iid = True
    # args.epochs = 20
    # args.lr = 0.01
    # args.model = 'resnet'
    # args.num_users = 5

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar100', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid:
            dict_users = cifar100_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR100')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # for bit in [args.quantization_bits]:
    for bit in [32,7]:
        mainFunction(bit)

