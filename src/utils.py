#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sampling import mnist_iid, mnist_noniid,cifar_iid, cifar_noniid, seed_iid, chbmit_iid

class MyDataset(Dataset):
    # read in two numpy array
    def __init__(self, data, label):
        super(MyDataset, self).__init__()
        self.data = data.astype(np.float32)
        self.label = label.astype(np.int64)
 
    def __len__(self):
        return self.label.shape[0]
 
    def __getitem__(self, item):
        return self.data[item], self.label[item]
    
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = 'dataset'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':

        data_dir = 'dataset'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
        print(type(train_dataset))

        print(train_dataset)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:

            user_groups = mnist_noniid(train_dataset, args.num_users)
    
    elif args.dataset == 'seed':
        data_dir = './dataset/seed_small/'

        test_id = args.target_id

        assert test_id > 0 and test_id <= 15

        test_data = standardization(np.load("./dataset/seed_small/" + str(test_id) + "_data.npy"))
        # print(test_data)
        test_label = np.load("./dataset/seed_small/" + str(test_id) + "_label.npy")
        test_label += 1



        train_data = []
        train_label = []
        for i in range(1, 16):
            if(i == test_id):
                continue
            x_data = standardization(np.load("./dataset/seed_small/" + str(i) + "_data.npy"))
            x_label = np.load("./dataset/seed_small/" + str(i) + "_label.npy")
            train_data.append(x_data)
            train_label.append(x_label)

        train_data = np.concatenate((train_data))
        train_label = np.concatenate((train_label))
        train_label += 1

        print("train_data.shape = ", train_data.shape)
        print("train_data.shape = ", train_label.shape)
        train_dataset = MyDataset(train_data, train_label)

        test_dataset = MyDataset(test_data, test_label)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = seed_iid(train_dataset, args.num_users)
        else:
            user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'chbmit':
        data_dir = './dataset/chbmit/'

        test_id = args.target_id

        assert test_id > 0 and test_id <= 10

        raw_data = np.load(data_dir + "de_data.npy")
        raw_label = np.load(data_dir + "chbmit_lable.npy")
        
        # this dataset contains 4 subtypes: arrange like [0,0,0,1,1,1,2,2,2,3,3,3]
        n = raw_data.shape[0]
        each_type_n = n // 4
        each_sub_n = (each_type_n // 10)

        train_data = []
        train_label = []
        for i in range(1, 11):
            x_data = [raw_data[each_sub_n*(i-1):each_sub_n*i]]
            x_data.append(raw_data[each_type_n + each_sub_n*(i-1) : each_type_n + each_sub_n*i])
            x_data.append(raw_data[each_type_n*2 + each_sub_n*(i-1) : each_type_n*2 + each_sub_n*i])  
            x_data.append(raw_data[each_type_n*3 + each_sub_n*(i-1) : each_type_n*3 + each_sub_n*i])
            x_label = [[0]*each_sub_n, [1]*each_sub_n, [2]*each_sub_n, [3]*each_sub_n]
            
            x_data = np.concatenate((x_data))
            x_label = np.concatenate((x_label))
            x_data = standardization(x_data)

            print("x_data.shape = ", x_data.shape)
            print("x_label.shape = ", x_label.shape)

            if i == test_id:
                test_data = x_data
                test_label = x_label
            else:
                train_data.append(x_data)
                train_label.append(x_label)

        train_data = np.concatenate((train_data))
        train_label = np.concatenate((train_label))

        print("train_data.shape = ", train_data.shape)
        print("train_data.shape = ", train_label.shape)
        train_dataset = MyDataset(train_data, train_label)

        test_dataset = MyDataset(test_data, test_label)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = chbmit_iid(train_dataset, args.num_users)
        else:
            user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def generate_averaged_test_data(test_dataset, sharing_ratio, aggregate_num):
    shared_data = []
    total_num = sharing_ratio * len(test_dataset)
    while (len(shared_data) < total_num):
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=aggregate_num,
                                                shuffle=True, drop_last=True, num_workers=4)
        for x, y in test_loader:
            x = torch.mean(x, dim=0, keepdim=True)
            shared_data.append(x)
            if(len(shared_data) >= total_num):
                break
    return torch.concat(shared_data)


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
