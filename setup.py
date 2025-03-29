#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from datetime import datetime
from time import strftime
import os
import time
import json

from client_split_sample.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from client_split_sample.Dirichlet_split_datasets import split_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CharLSTM
from models.Fed import FedWeightAvg, weight_agg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from utils.utils import exp_details, worst_fraction, best_fraction, add_noise


if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.method = "setup"
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)  #训练集
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)  #测试集
        # sample users
        if args.iid:
            print("iid")
            dict_users = mnist_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        else:
            if args.dirichlet != 0:
                print("non-iid->dirichlet")
                labels_train = np.array(dataset_train.targets)
                dict_users = split_noniid(labels_train, args.dirichlet, args.num_users)
            else:
                print("non-iid->shard")
                dict_users = mnist_noniid(dataset_train, args.num_users)  # 否则为用户分配non-iid数据
        
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)  #训练集
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)  #测试集
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        else:
            if args.dirichlet != 0:
                print("dirichlet")
                labels_train = np.array(dataset_train.targets)
                dict_users = split_noniid(labels_train, args.dirichlet, args.num_users)
            else:
                print("shard")
                dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')

    # print(dict_users)
    new_dict_users = {}
    for key,value in dict_users.items():
        new_dict_users[key] = [int(x) for x in value]
    # print(new_dict_users)
    with open('./setup/dict_users.json', 'w') as f:
        f.truncate(0)
        json.dump(new_dict_users, f)

    print(dataset_train)
    

    if args.dataset == 'mnist':
        img_size = [1,28,28]
    elif args.dataset == 'cifar':
        img_size = [3,32,32]
    # print(img_size)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    print('Number of clients: {:3d}'.format(args.num_users))

    torch.save(net_glob, './setup/model.pth')

    # loaded_model = torch.load('./setup/model.pth')

    