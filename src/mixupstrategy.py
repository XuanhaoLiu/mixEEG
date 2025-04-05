import torch
from torch import nn
import random
import numpy as np

def Linear_mixup(x_a, x_b, type=1.0):
    assert(x_a.device == x_b.device)

    alpha = type
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_a.size()[0]
    index = torch.randperm(batch_size)
    # print(index)
    # print(index.shape)

    mixed_x = lam * x_a + (1 - lam) * x_b[index, :]
    return mixed_x, lam

def Channel_mixup(x_a, x_b, type):
    assert(type == "random" or type == "binary")

    channel_number = x_a.size()[1]

    if type == "random":
        half_channel = torch.randperm(channel_number)[:channel_number//2]
        a_mask = torch.zeros(channel_number)
        a_mask[half_channel] = 1
        b_mask = torch.ones_like(a_mask) - a_mask
    else:
        # a_mask contians all channels from the left side of EEG
        if(channel_number == 62):
            a_mask = torch.tensor([1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,])
        elif(channel_number == 23):
            # Channel 1: FP1-F7 Channel 2: F7-T7 Channel 3: T7-P7 Channel 4: P7-O1
            # Channel 5: FP1-F3 Channel 6: F3-C3 Channel 7: C3-P3 Channel 8: P3-O1
            # Channel 9: FP2-F4 Channel 10: F4-C4 Channel 11: C4-P4 Channel 12: P4-O2
            # Channel 13: FP2-F8 Channel 14: F8-T8 Channel 15: T8-P8 Channel 16: P8-O2
            # Channel 17: FZ-CZ Channel 18: CZ-PZ Channel 19: P7-T7 Channel 20: T7-FT9
            # Channel 21: FT9-FT10 Channel 22: FT10-T8 Channel 23: T8-P8
            a_mask = torch.tensor([1,1,1,1, 1,1,1,1, 0,0,0,0, 0,0,0,0, 1,0, 1,1, 1,0,0])
        b_mask = torch.ones_like(a_mask) - a_mask
        # randomly swap a_mask and b_mask for balancing the data mixup,
        # i.e., we don't want to always use a's left side and b's right side
        if random.random() < 0.5:
            a_mask, b_mask = b_mask, a_mask
    a_mask = a_mask.to(x_a.device)
    b_mask = b_mask.to(x_b.device)
    mixed_x = x_a * a_mask.view(1, channel_number, 1) + x_b * b_mask.view(1, channel_number, 1)
    return mixed_x, 0.5

def Frequency_mixup(x_a, x_b, type):
    # cut means cut the frequency band {a,b,c,d,e} into {a,b,c} and {d,e}
    # cross means cut the frequency band {a,b,c,d,e} into {a,c,e} and {b,d}
    assert(type == "cut" or type == "cross")
    if type == "cut":
        a_mask = torch.tensor([1,1,1,0,0])
    else:
        a_mask = torch.tensor([1,0,1,0,1])

    b_mask = torch.ones_like(a_mask) - a_mask
    lam = 0.6
    if random.random() < 0.5:
        a_mask, b_mask = b_mask, a_mask
        lam = 0.4
    a_mask = a_mask.to(x_a.device)
    b_mask = b_mask.to(x_b.device)
    mixed_x = x_a * a_mask.view(1, 1, 5) + x_b * b_mask.view(1, 1, 5)
    return mixed_x, lam

if __name__ == "__main__":

    n = 1
    x_a = torch.rand(size=(n, 62, 5))
    x_b = torch.rand(size=(n, 62, 5))

    # n_classes = 2
    # y_a = nn.functional.one_hot(torch.tensor(np.array([1]*n)), n_classes).float()
    # y_b = nn.functional.one_hot(torch.tensor(np.array([0]*n)), n_classes).float()

    # mixed_x, lam = Channel_mixup(x_a, x_b, type="binary")

    mixed_x, lam = Frequency_mixup(x_a, x_b, type="cross")

    print(mixed_x)
    print(lam)