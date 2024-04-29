import torch.nn as nn
import torch

#-----------------------------------------------------------
#-----------------------------------------------------------
# BN for the input seed 
#-----------------------------------------------------------
#-----------------------------------------------------------
class BNNet(nn.Module):
    def __init__(self,num_channel):
        super(BNNet, self).__init__()
        self.bn = nn.BatchNorm2d(num_channel)

    def forward(self, input_data):
        output_data = self.bn(input_data)
        return output_data


def tv1_loss(x):
    #### here, our input must be {batch, channel, height, width}
    ndims = len(list(x.size()))
    if ndims != 4:
        assert False, "Input must be {batch, channel, height, width}"
    n_pixels = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    tot_var = torch.sum(dh) + torch.sum(dw)
    tot_var = tot_var / n_pixels
    return tot_var
