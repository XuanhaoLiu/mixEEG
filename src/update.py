#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
from mixupstrategy import Linear_mixup, Channel_mixup, Frequency_mixup

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size=self.args.local_bs, shuffle=True, num_workers=4)

    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                output = model(images)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()

        torch.cuda.empty_cache()
        return model.state_dict()
    
    def update_weights_mixup(self, model, mixup_s, alpha, subtype):
        assert mixup_s in ["lin", "cha", "fre"]
        if(mixup_s == "lin"):
            mixup = Linear_mixup
            mixup_para = alpha
        elif(mixup_s == "cha"):
            mixup = Channel_mixup
            mixup_para = subtype
        else:
            mixup = Frequency_mixup
            mixup_para = subtype

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(), labels.cuda()
                
                shuffle_idx = torch.randperm(images.shape[0]).cuda()
                images_b, labels_b = images[shuffle_idx, :], labels[shuffle_idx]

                mixed_images, lam = mixup(images, images_b, type=mixup_para)

                model.zero_grad()
                output = model(mixed_images)
                loss = mixup_criterion(F.cross_entropy, output, labels, labels_b, lam)
                # loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                
        torch.cuda.empty_cache()
        return model.state_dict()

class LocalUpdate_DA(object):
    def __init__(self, args, dataset, idxs, shared_data):
        self.args = args
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        self.shared_data = shared_data
        self.sharedloader = DataLoader(shared_data, batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        self.n_class = 3
        if args.dataset == "chbmit":
            self.n_class += 1

    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                output = model(images)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()

            for batch_idx, images in enumerate(self.sharedloader):
                images = images.cuda()
                model.zero_grad()
                output = model(images)
                loss = 0
                for i in range(self.n_class):
                    loss += F.cross_entropy(output, torch.tensor([i] * images.shape[0]).long().cuda())
                loss /= self.n_class
                loss.backward()
                optimizer.step()

        torch.cuda.empty_cache()
        return model.state_dict()
    
    def update_weights_mixup(self, model, mixup_s, alpha, subtype):
        assert mixup_s in ["lin", "cha", "fre"]
        if(mixup_s == "lin"):
            mixup = Linear_mixup
            mixup_para = alpha
        elif(mixup_s == "cha"):
            mixup = Channel_mixup
            mixup_para = subtype
        else:
            mixup = Frequency_mixup
            mixup_para = subtype

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)

        shared_data = self.shared_data.cuda()

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(), labels.cuda()
                
                shuffle_idx = torch.randperm(images.shape[0]).cuda()
                images_b, labels_b = images[shuffle_idx, :], labels[shuffle_idx]

                mixed_images, lam = mixup(images, images_b, type=mixup_para)

                model.zero_grad()
                output = model(mixed_images)
                loss = mixup_criterion(F.cross_entropy, output, labels, labels_b, lam)
                # loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(), labels.cuda()

                indices = torch.randperm(shared_data.shape[0])[:images.shape[0]].cuda()
                picked_shared_data = shared_data[indices]

                mixed_images, lam = mixup(images, picked_shared_data, type=mixup_para)

                model.zero_grad()
                output = model(mixed_images)
                
                loss = lam * F.cross_entropy(output, labels)
                loss_mix = 0
                for i in range(self.n_class):
                    loss_mix += F.cross_entropy(output, torch.tensor([i] * images.shape[0]).long().cuda())
                loss_mix /= self.n_class
                loss += ((1 - lam) * loss_mix)
                # loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                
        torch.cuda.empty_cache()
        return model.state_dict()
