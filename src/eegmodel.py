import torch
from torch import nn

################################### EEG Model for FedAvg ####################################
# Time : 2024.12.25
# Author : Xuanhao Liu

# These EEG Models contains two types: MLP and CNN, we test these models
#############################################################################################

class MLP(nn.Module):
    def __init__(self, classes_num, adapt_dim = 100):
        super(MLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(adapt_dim)
        self.net = nn.Sequential(
            nn.Linear(adapt_dim, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(64, classes_num)
    
    def forward(self, x, output_feature=False):               #input:(batch,C,5)
        # print("input data shape:", x.shape)
        x = self.pool(nn.Flatten()(x))
        # print("after pooling shape:", x.shape)
        x = self.net(x)             
        if output_feature:
            return x
        return self.out(x)

class CNN(nn.Module):
    def __init__(self, classes_num, adapt_dim = 100):
        super(CNN, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(adapt_dim)
        self.net = nn.Sequential(
            nn.Conv2d(
               in_channels=1,
               out_channels=16,     
               kernel_size=(3, 3),     # filter size
            ),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Conv2d(
               in_channels=16,
               out_channels=16,     
               kernel_size=(3, 3),     # filter size
            ),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Conv2d(
               in_channels=16,
               out_channels=16,     
               kernel_size=(7, 1),     # filter size
            ),
            nn.GELU(),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(adapt_dim, classes_num)
    
    def forward(self, x, output_feature=False):               #input:(batch,C,5)
        x = x.unsqueeze(1)
        x = self.net(x)         
        # print(x.shape)
        x = self.pool(nn.Flatten()(x))
        if output_feature:
            return x
        return self.out(x)
    
def get_net(net_type, classes_num, adapt_dim):         #train.py will call this function to get the net
    assert(net_type == "CNN" or net_type == "MLP")
    if net_type == "CNN":
        model = CNN(classes_num, adapt_dim)
    elif net_type == "MLP":
        model = MLP(classes_num, adapt_dim)
    return model

if __name__ == "__main__":
    model = get_net("MLP", 3, 100)
    x = torch.rand(size=(32, 62, 5))
    print(x.shape)
    y = model(x)
    print(y.shape)  #if input(b,1,1,3000),then the output is(1,num_classes)