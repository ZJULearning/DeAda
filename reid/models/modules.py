from __future__ import print_function, division
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function

class DetSelection(nn.Module):
    
    def __init__(self, channel_num, feature_h, feature_w):
        super(DetSelection, self).__init__()
        self.channel_num = channel_num
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.weight = nn.Parameter(torch.Tensor(channel_num, feature_h, feature_w))
        
        self.reset_parameters()

    def reset_parameters(self):
        # to make init sigmoid close to 1
        #self.weight.data.uniform_(5, 10)
        self.weight.data.fill_(5)

    def forward(self, input):
        mask = torch.sigmoid(self.weight)
        output = input.mul(mask.expand_as(input))
        # return mask for visualization
        return mask, output


class GumbelSample(nn.Module):

    def __init__(self, init_tau=1.0, use_sigmoid=False, use_reg_max=False):
        super(GumbelSample, self).__init__()
        self.tau = init_tau
        self.use_sigmoid = use_sigmoid
        self.use_reg_max = use_reg_max

    def forward(self, input):
        assert input.dim() == 4
        batch_size = input.size()[0]
        K = 2    # active or not active  1 means active  2 means not active
        N = self.num_flat_features(input)
        #print(batch_size, N, M)
        input_size = input.size()

        """ there are three initial methods; one is sigmoid, another is use max() min() for each channel """
        
        if self.use_sigmoid == True:
            input = torch.sigmoid(input)
        else:
            input = input.view(input_size[0] * input_size[1], input_size[2] * input_size[3])
            input_max, _ = input.max(-1)
            input_max.detach_()
            input_min, _ = input.min(-1)
            input_min.detach_()

            #assert (input_max > input_min).data.size()[0] == (input_max > input_min).data.sum()

            input = (input - input_min.unsqueeze(1).expand_as(input)) / ((input_max - input_min + 1e-20).unsqueeze(1).expand_as(input))
            
            if self.use_reg_max:
                mul = Variable(torch.min(input_max.data, torch.ones(input_max.size()).cuda()))
                input = input * (mul.unsqueeze(1).expand_as(input))
            
            input = input.view(input_size[0], input_size[1], input_size[2], input_size[3])
        
        """ now begin to gumbel_softmax """
        
        logits_y1 = torch.unsqueeze(input, -1)
        logits_y2 = 1 - logits_y1
        logits_y = torch.cat((logits_y1, logits_y2), -1)
        logits_y = logits_y.view(-1, K)    

        prob = self.gumbel_softmax_sample(logits_y)
        prob = prob.view(input_size[0], input_size[1], input_size[2], input_size[3], K)
        mask = prob[:,:,:,:, 0]        # 0 means the probability of choose,  1 means no choose
        assert mask.size() == input.size()

        return mask

    def adjust_tau(self, tau):
        #print("######### origin tau is ", self.tau)
        self.tau = tau
        #print("######## new tau is ", self.tau)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
   
    def gumbel_softmax_sample(self, logits, eps=1e-20):
        assert logits.dim() == 2
        """ Sample from Gumbel(0, 1) """
        U = torch.Tensor(logits.size()).cuda()
        U.uniform_(0, 1)
        gumbel = Variable(-torch.log(-torch.log(U + eps) + eps))
        
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = (logits + gumbel) / self.tau
        return F.softmax(y) 
        




