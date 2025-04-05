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
from update import LocalUpdate, LocalUpdate_DA
from models import CNNMnist, CNNCifar
from eegmodel import get_net
from utils import get_dataset, average_weights, exp_details, generate_averaged_test_data
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def inference(model, test_loader, dataset):
    if dataset == "seed":
        num_classes = 3
    elif dataset == "chbmit":
        num_classes = 4
    model.eval()
    test_loss = 0.0
    correct = 0.0
    all_pred = []
    all_target = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_pred += pred.tolist()
            all_target += target.tolist()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    acc_ = 100. * metrics.precision_score(all_target, all_pred, average='micro')
    f1_score = 100. * metrics.f1_score(all_target, all_pred, average='weighted')
    classes=[i for i in range(num_classes)]
    auc_roc = 100. * metrics.roc_auc_score(label_binarize(all_target, classes=classes), label_binarize(all_pred, classes=classes), multi_class='ovr')
    kappa = 100. * metrics.cohen_kappa_score(all_target, all_pred)

    print(acc, acc_)
    print("f1_score = ", f1_score)
    print("auc_roc = ", auc_roc)
    print("kappa_score = ", kappa)

    return acc_, f1_score, auc_roc, kappa, test_loss


def prepare_folders(cur_path):
    folders_util = [
        os.path.join(cur_path + '/logs', args.store_name),
        os.path.join(cur_path + '/checkpoints', args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


def save_checkpoint(state, is_best):
    filename = '{}/{}/ckpt.pth.tar'.format(os.path.abspath(os.path.dirname(os.getcwd())) + '/checkpoints',
                                           args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


if __name__ == '__main__':

    args = args_parser()
    args.type = 'iid' if args.iid == 1 else 'non-iid'
    args.store_name = '_'.join(
        ["AB_rs_LOSO_DA_", args.dataset, args.model, args.type, args.mixup_strategy, 'a-' + str(args.mixup_alpha), args.mixup_subtype, 'lr-' + str(args.lr)])
    cur_path = os.path.abspath(os.path.dirname(os.getcwd()))
    print("cur_path = ", cur_path)
    prepare_folders(cur_path)
    exp_details(args)

    logger_file = open(os.path.join(cur_path + '/logs', args.store_name, 'log.txt'), 'w')
    tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + '/logs', args.store_name))

    # load dataset and user groups

    transfer_target_bst_acc = []
    transfer_target_bst_f1 = []
    transfer_target_bst_aucroc = []
    transfer_target_bst_kappa = []

    if args.dataset == "seed":
        subject_num = 15
    elif args.dataset == "chbmit":
        subject_num = 10

    for r in range(1, 11):
        for s in range(1, 11):
        
            args.r = r / 100.0
            args.aggregate = s * 5

            train_dataset, test_dataset, user_groups = get_dataset(args)

            shared_data = generate_averaged_test_data(test_dataset, sharing_ratio=args.r, aggregate_num=args.aggregate)
            print("shared_data = ", shared_data.shape)

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

            bst_acc, bst_f1, bst_aucroc, bst_kappa = -1, -1, -1, -1
            description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
            for epoch in tqdm(range(args.epochs)):
                local_weights = []
                global_model.train()
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                print(f"epoch = {epoch}, idxs_users = {idxs_users}")

                for idx in idxs_users:
                    local_model = LocalUpdate_DA(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], shared_data=shared_data)
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

                test_acc, f1_score, auc_roc, kappa, test_loss = inference(global_model, test_loader, args.dataset)

                tf_writer.add_scalar('test_acc', test_acc, epoch)
                tf_writer.add_scalar('test_loss', test_loss, epoch)

                output_log = 'After {} global rounds, Test acc: {}, inference loss: {}'.format(
                    epoch + 1, test_acc, test_loss)

                logger_file.write(output_log + '\n')
                logger_file.flush()

                is_best = test_acc > bst_acc
                if is_best:
                    bst_acc = max(bst_acc, test_acc)
                    bst_f1 = f1_score
                    bst_aucroc = auc_roc
                    bst_kappa = kappa
                print(description.format(test_acc, test_loss, bst_acc))
                
                save_checkpoint(global_model.state_dict(), is_best)
            transfer_target_bst_acc.append(bst_acc)
            transfer_target_bst_f1.append(bst_f1)
            transfer_target_bst_aucroc.append(bst_aucroc)
            transfer_target_bst_kappa.append(bst_kappa)
            # break
    
    transfer_target_bst_acc = np.array(transfer_target_bst_acc)
    transfer_target_bst_f1 = np.array(transfer_target_bst_f1)
    transfer_target_bst_aucroc = np.array(transfer_target_bst_aucroc)
    transfer_target_bst_kappa = np.array(transfer_target_bst_kappa)
    savedata = np.concatenate((transfer_target_bst_acc[np.newaxis, :], transfer_target_bst_f1[np.newaxis, :]), axis=0)
    savedata = np.concatenate((savedata, transfer_target_bst_aucroc[np.newaxis, :]), axis=0)
    savedata = np.concatenate((savedata, transfer_target_bst_kappa[np.newaxis, :]), axis=0)
    np.save("./bst_acc/" + args.store_name + ".npy", savedata)
    print("acc_mean = ", np.mean(transfer_target_bst_acc), np.std(transfer_target_bst_acc))
    print("f1_mean = ", np.mean(transfer_target_bst_f1), np.std(transfer_target_bst_f1))
    print("auc_roc_mean = ", np.mean(transfer_target_bst_aucroc), np.std(transfer_target_bst_aucroc))
    print("kappa_mean = ", np.mean(transfer_target_bst_kappa), np.std(transfer_target_bst_kappa))

"""
python3 federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=300 --lr=0.01 --local_ep=5 --local_bs=32

python3 federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32

"""