# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import skimage.io
from skimage.transform import resize
import numpy as np

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:

        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #net.to(gpu_ids[0])
        net.cuda()
        net = torch.nn.DataParallel(net)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'AutoEncoder':
        net = AutoEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

class AutoEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d):
        super(AutoEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        self.filters = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # ---
        model01 = [nn.Conv2d(3, ngf*4, kernel_size=5, padding=2, stride=2,
                             bias=use_bias),
                   norm_layer(ngf*4),
                   nn.ReLU(0.2)]
        model02 = [nn.Conv2d(ngf*4, ngf*8, kernel_size=5,
                             stride=2, padding=2, bias=use_bias),
                   norm_layer(ngf*8),
                   nn.ReLU(0.2)]
        model03 = [nn.Conv2d(ngf*8, ngf*12, kernel_size=5,
                             stride=2, padding=2, bias=use_bias),
                   norm_layer(ngf*12),
                   nn.ReLU(0.2)]
        model03 += [nn.Conv2d(ngf*12, ngf*16, kernel_size=5,
                              stride=2, padding=2, bias=use_bias),
                    norm_layer(ngf*16),
                    nn.ReLU(0.2)]
        model04 = [nn.Conv2d(ngf*16, ngf*16, kernel_size=3,
                             stride=1, padding=1, bias=use_bias),
                   norm_layer(ngf*16),
                   nn.ReLU(0.2),
                   nn.MaxPool2d(2, stride=2),
                   norm_layer(ngf*16),
                   nn.ReLU(0.2)]
        model04 += [nn.Conv2d(ngf*16, ngf*12, kernel_size=3,
                              stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf*12),
                    nn.ReLU(0.2),
                    nn.MaxPool2d(2, stride=2),
                    norm_layer(ngf*12),
                    nn.ReLU(0.2)]

        model2 = [nn.ConvTranspose2d(ngf*12, ngf*16, kernel_size=4,
                                     stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf*16),
                  nn.ReLU(True)]
        model2 += [nn.ConvTranspose2d(ngf*16, ngf*16, kernel_size=4,
                                      stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf*16),
                   nn.ReLU(True)]
        model2 += [nn.ConvTranspose2d(ngf*16, ngf*16, kernel_size=4,
                                      stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf*16),
                   nn.ReLU(True)]
        model2 += [nn.ConvTranspose2d(ngf*16, ngf*12, kernel_size=4,
                                      stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf*12),
                   nn.ReLU(True)]
        model2 += [nn.ConvTranspose2d(ngf*12, ngf*8, kernel_size=4,
                                      stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf*8),
                   nn.ReLU(True)]
        model2 += [nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4,
                                      stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf*4),
                   nn.ReLU(True)]
        model2 += [nn.ConvTranspose2d(ngf*4, ngf, kernel_size=4,
                                      stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf),
                   nn.ReLU(True)]
        model2 += [nn.Conv2d(ngf, 3, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
                   nn.Tanh()]

        self.model01 = nn.Sequential(*model01)
        self.model02 = nn.Sequential(*model02)
        self.model03 = nn.Sequential(*model03)
        self.model04 = nn.Sequential(*model04)
        self.model2 = nn.Sequential(*model2)

    def forward(self, x, cond):
        if cond == 0:
            enc = self.model04(self.model03(self.model02(self.model01(x))))
            dec = self.model2(enc)
            return dec

        if cond == 1:
            enc = self.model04(self.model03(self.model02(self.model01(x))))
            return enc

        if cond == 2:
            dec = self.model2(x)
            return dec

        if cond == 3:
            f01 = self.model01(x)
            f02 = self.model02(f01)
            f03 = self.model03(f02)
            f04 = self.model04(f03)
            return f01, f02, f03, f04

