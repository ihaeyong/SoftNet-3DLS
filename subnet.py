import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
import math
import numpy as np

def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()

class GetSubnetFaster(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity, sub_type):
        with torch.no_grad():
            if sub_type == 'softnet':
                zeros = torch.rand_like(zeros)
            k_val = percentile(scores.abs(), sparsity*100)
            onehot = torch.where(scores.abs() < k_val,
                                 zeros.to(scores.device),
                                 ones.to(scores.device))

            ctx.save_for_backward(ones.to(scores.device))
        return onehot

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None, None

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def get_none_masks(model):
    none_masks = {}
    for name, module in model.named_modules():
        if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
            none_masks[name + '.weight'] = None
            none_masks[name + '.bias'] = None

class SignetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, trainable=True, sub_type='hardnet'):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = 0.3
        self.trainable = trainable

        assert sub_type in ['softnet', 'hardnet']
        self.sub_type = sub_type

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if self.training:
            if weight_mask is None:
                self.weight_mask = GetSubnetFaster.apply(self.w_m,
                                                         self.zeros_weight,
                                                         self.ones_weight,
                                                         self.sparsity,
                                                         self.sub_type)
            else:
                self.weight_mask = weight_mask
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = self.sigmoid(self.b_m)
                b_pruned = self.bias_mask * self.bias

        else:
            if weight_mask is None:
                weight_mask = GetSubnetFaster.apply(self.w_m,
                                                    self.zeros_weight,
                                                    self.ones_weight,
                                                    self.sparsity,
                                                    self.sub_type)
            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

class SignetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, trainable=True):
        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.stride = stride
        self.sparsity = 0.1
        self.trainable = trainable
        self.noise = False
        self.view_id = 0
        self.view_scale = 1.0

        # Mask Parameters of Weight and Bias
        self.w_m = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))

        self.scores = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size)),
                nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
            ])


        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters(self.w_m)
        self.init_mask_parameters(self.scores[0])
        self.init_mask_parameters(self.scores[1])

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):

        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if self.training:
            if weight_mask is None:
                if False:
                    self.weight_mask = GetSubnetFaster.apply(
                        self.scores,
                        self.zeros_weight,
                        self.ones_weight,
                        self.sparsity,
                        self.view_id, self.view_scale)
                else:
                    self.weight_mask = GetSubnetFasterSVD.apply(
                        self.w_m,
                        self.zeros_weight,
                        self.ones_weight,
                        self.sparsity,
                        self.view_scale)


            else:
                self.weight_mask = weight_mask

            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = self.b_m
                b_pruned = self.bias_mask * self.bias

        # If inference/test, no need to compute the subnetwork
        else:
            if weight_mask is None:
                if False:
                    weight_mask = GetSubnetFaster.apply(
                        self.scores,
                        self.zeros_weight,
                        self.ones_weight,
                        self.sparsity,
                        self.view_id, self.view_scale)
                else:
                    weight_mask = GetSubnetFasterSVD.apply(
                        self.w_m,
                        self.zeros_weight,
                        self.ones_weight,
                        self.sparsity,
                        self.view_scale)

            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def init_mask_parameters(self, w_m):
        nn.init.kaiming_uniform_(w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

        # set initial weight
        self.w_m_init = self.w_m.detach()

