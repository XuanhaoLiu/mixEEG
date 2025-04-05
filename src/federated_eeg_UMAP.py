#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import os
import shutil
import warnings

import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate
from models import CNNMnist, CNNCifar
from eegmodel import get_net
from utils import get_dataset, average_weights, exp_details
import torch.nn.functional as F


def inference(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            print(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc, test_loss

def inference_getfeature(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    features = 1
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data, True)
            if type(features) is np.ndarray:
                features = np.concatenate((features, output.cpu().numpy()))
            else:
                features = output.cpu().numpy()

    print(features.shape)
    return features

def prepare_folders(cur_path):
    folders_util = [
        os.path.join(cur_path + '/logs', args.store_name),
        os.path.join(cur_path + '/checkpoints', args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


def save_checkpoint(state, is_best):
    filename = '{}/UMAP_{}/ckpt.pth.tar'.format(os.path.abspath(os.path.dirname(os.getcwd())) + '/checkpoints',
                                           args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


if __name__ == '__main__':

    args = args_parser()
    args.type = 'iid' if args.iid == 1 else 'non-iid'
    args.store_name = '_'.join(
        [args.dataset, args.model, args.type, args.mixup_strategy, 'a-' + str(args.mixup_alpha), args.mixup_subtype, 'lr-' + str(args.lr)])
    cur_path = os.path.abspath(os.path.dirname(os.getcwd()))
    print("cur_path = ", cur_path)
    prepare_folders(cur_path)
    exp_details(args)

    logger_file = open(os.path.join(cur_path + '/logs', args.store_name, 'log.txt'), 'w')
    tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + '/logs', args.store_name))

    # load dataset and user groups

    
    train_dataset, test_dataset, user_groups = get_dataset(args)

    for i in user_groups:
        print(i)
    # print(user)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                              shuffle=False, num_workers=4)
    # BUILD MODEL
    if args.dataset == 'seed':
        if(args.model == 'mlp'):
            global_model = get_net("MLP", 3, 100).cuda()
        else:
            global_model = get_net("CNN", 3, 100).cuda()
    elif args.dataset == 'chbmit':
        if(args.model == 'mlp'):
            global_model = get_net("MLP", 4, 100).cuda()
        else:
            global_model = get_net("CNN", 4, 100).cuda()

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    bst_model_stat = None
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        print(f"epoch = {epoch}, idxs_users = {idxs_users}")

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            if args.mixup_strategy == "none":
                w = local_model.update_weights(
                    model=copy.deepcopy(global_model))
            else:
                w = local_model.update_weights_mixup(
                    model=copy.deepcopy(global_model),
                    mixup_s=args.mixup_strategy,
                    alpha=args.mixup_alpha, subtype=args.mixup_subtype)
            local_weights.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        test_acc, test_loss = inference(global_model, test_loader)

        tf_writer.add_scalar('test_acc', test_acc, epoch)
        tf_writer.add_scalar('test_loss', test_loss, epoch)

        output_log = 'After {} global rounds, Test acc: {}, inference loss: {}'.format(
            epoch + 1, test_acc, test_loss)

        logger_file.write(output_log + '\n')
        logger_file.flush()

        is_best = test_acc > bst_acc
        bst_acc = max(bst_acc, test_acc)
        print(description.format(test_acc, test_loss, bst_acc))
        
        # save_checkpoint(global_model.state_dict(), is_best)
        if is_best:
            bst_model_stat = global_model.state_dict()

    global_model.load_state_dict(bst_model_stat)
    learned_features = inference_getfeature(global_model, test_loader)
    np.save("./LearnedFeature/" + args.store_name + ".npy", learned_features)
"""
python3 federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=300 --lr=0.01 --local_ep=5 --local_bs=32

python3 federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32

"""