import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self,img_shape,z_dim):
        super(Generator, self).__init__()
        self.img_rows,self.img_cols,self.channels = img_shape
        self.hidden_size = 128
        self.linear1 = nn.Linear(z_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.img_rows*self.img_cols*self.channels)

    def forward(self,x):
        # x = F.leaky_relu(self.linear1(x))
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        # x = x.view(-1, self.channels, self.img_rows, self.img_cols)
        return x

class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()
        self.img_rows,self.img_cols,self.channels = img_shape
        self.hidden_size = 128
        self.linear1 = nn.Linear(self.img_rows*self.img_cols*self.channels, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)

    def forward(self,x):
        # x = x.view(-1, self.img_rows*self.img_cols*self.channels)
        x = F.leaky_relu(self.linear1(x),0.2)
        x = F.leaky_relu(self.linear2(x),0.2)
        x = torch.sigmoid(self.linear3(x))
        return x
