import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from utils import conv_norm

class TrimDense(Module):
    """
    Dense layer for the Trimmed \ell_1 Regularization
    We treat the network parameters as a Bayesian
    """
    def __init__(self, in_features, out_features, log_alpha='hidden', lamda=0.1, h=0, bias=True):
        """
        Args:
            in_features:    the number of input-neurons
            out_features:   the number of output-neurons
            h:              the number of largest entries which do not be penalized
            bias:           use bias or not
        """
        super(TrimDense, self).__init__()
        assert in_features >= h

        # Fully-Connected Layers
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # For Bayesian Inference
        self.log_alpha = Parameter(torch.randn(in_features, out_features).fill_(-10.0))
        self.c1 = 1.16145124
        self.c2 = -1.50204118
        self.c3 = 0.58629921
        
        # Trimming Parameters
        self.lamda = lamda                              # regularization parameters
        self.n_pnt = in_features - h                    # the number of penalties
        
        if torch.cuda.is_available():
            self.regw = torch.ones(in_features).cuda()  # input-neuron sparsity
        else:
            self.regw = torch.ones(in_features)
        
        self.mask = None
        self.bias_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data, mode='fan_out')
        if self.bias is not None:
            self.bias.data.fill_(0)
        self.regw.data.fill_(self.n_pnt/self.in_features)
        self.regw.requires_grad_()

    def forward(self, inputs):
        self.log_alpha.data.clamp_(max=0.0)
        if self.training:
            # Local Reparametrization Trick on Training Phase
            mu = torch.mm(inputs, self.weight)
            std = torch.sqrt(torch.mm(inputs**2, self.log_alpha.exp() * self.weight**2) + 1e-8)
            
            # This means that sampling only one time for each datapoint
            eps = torch.randn(*mu.size())
            if inputs.is_cuda:
                eps = eps.cuda()
            return std * eps + mu + self.bias
        else:
            # Test Phase
            if self.mask is not None:
                return torch.addmm(self.bias, inputs, self.mask * self.weight)
            else:
                return torch.addmm(self.bias, inputs, self.weight)

    def kld(self):
        """
        Variational Dropout and the Local Reparametrization Trick
        --> Variational (A2) method
        """
        self.log_alpha.data.clamp_(max=0.0)
        alpha = self.log_alpha.exp()
        nkld = 0.5 * self.log_alpha + self.c1 * alpha + self.c2 * alpha**2 + self.c3 * alpha**3
        kld = -nkld
        return kld.mean() / 3 

    def compute_expected_flops(self):
        """
        To be implemented
        """
        return
    
    def reg_theta_loss(self, batch_size=100):
        """
        For subgradient method
        """
        eps = torch.randn(*self.weight.size())
        if torch.cuda.is_available():
            eps = eps.cuda()
        mu = self.weight 
        std = torch.sqrt(self.log_alpha.exp() * self.weight**2 + 1e-8) 
        samples = mu + std * eps
        
        return self.lamda * torch.sum(self.regw.data * samples.norm(dim=1))

    def reg_w_loss(self, batch_size=100):
        eps = torch.randn(*self.weight.size())
        if torch.cuda.is_available():
            eps = eps.cuda()
        mu = self.weight.data
        std = torch.sqrt(self.log_alpha.data.exp() * self.weight.data**2 + 1e-8)
        samples = mu + std * eps
        return torch.sum(self.regw * samples.norm(dim=1))

    def set_weight_mask(self):
        self.mask = (self.regw.data < 1.0).unsqueeze(-1).repeat(1, self.out_features)
        if torch.cuda.is_available():
            self.mask = self.mask.type(torch.cuda.FloatTensor)
        else:
            self.mask = self.mask.type(torch.FloatTensor)

    def set_bias_mask(self, next_layer_regw):
        assert len(next_layer_regw) == self.out_features
        self.bias_mask = next_layer_regw.data < 1.0
        if torch.cuda.is_available():
            self.bias_mask = self.bias_mask.type(torch.cuda.FloatTensor)
        else:
            self.bias_mask = self.bias_mask.type(torch.FloatTensor)
        return

    def apply_mask(self):
        self.weight.data = self.weight.data * self.mask
        if self.bias_mask is not None:
            self.bias.data = self.bias.data * self.bias_mask
        return

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

class TrimConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 log_alpha=-10.0, lamda=0.1, h=0):
        super(TrimConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # For Bayesian inference
        self.log_alpha = Parameter(torch.randn(*self.weight.size()).fill_(log_alpha))
        
        # KL divergence
        self.c1 = 1.16145124
        self.c2 = -1.50204118
        self.c3 = 0.58629921
        
        # Trimming Parameters
        self.lamda = lamda                              # regularization parameters
        self.n_pnt = out_channels - h                   # the number of penalties
        
        if torch.cuda.is_available():
            self.regw = torch.ones(out_channels).cuda() # output feature map sparsity
        else:
            self.regw = torch.ones(out_channels)
        
        self.mask = None
        self.bias_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in')
        if self.bias is not None:
            self.bias.data.fill_(0.0)
        self.regw.data.fill_(self.n_pnt / self.out_channels)
        self.regw.requires_grad_()
    
    def forward(self, inputs):
        self.log_alpha.data.clamp_(max=0.0)
        if self.training:
            # Local Reparametrization Trick on Training Phase
            mu = F.conv2d(inputs, self.weight)
            std = torch.sqrt(F.conv2d(inputs**2, self.log_alpha.exp() * self.weight**2) + 1e-8)
            
            # This means that sampling only one time for each datapoint
            eps = torch.randn(*mu.size())
            if inputs.is_cuda:
                eps = eps.cuda()

            output_size = eps.size()
            conv_bias = self.bias.unsqueeze(0).repeat(output_size[0], 1).unsqueeze(-1).repeat(1, 1, output_size[2]).unsqueeze(-1).repeat(1, 1, 1, output_size[3])
            return torch.add((mu + std * eps), conv_bias)
        else:
            # Test Phase
            if self.mask is not None:
                return F.conv2d(inputs, self.mask * self.weight, self.bias)
            else:
                return F.conv2d(inputs, self.weight, self.bias)

    def kld(self):
        """
        Variational Dropout and the Local Reparametrization Trick
        --> Variational (A2) method
        """
        self.log_alpha.data.clamp_(max=0.0)
        alpha = self.log_alpha.exp()
        nkld = 0.5 * self.log_alpha + self.c1 * alpha + self.c2 * alpha**2 + self.c3 * alpha**3
        kld = -nkld
        return kld.mean() / 3 

    def compute_expected_flops(self):
        """
        To be implemented
        """
        return
    
    def reg_theta_loss(self):
        """
        For subgradient method
        """
        eps = torch.randn(*self.weight.size())
        if torch.cuda.is_available():
            eps = eps.cuda()
        mu = self.weight 
        std = torch.sqrt(self.log_alpha.exp() * self.weight**2 + 1e-8) 
        samples = mu + std * eps
        return self.lamda * torch.sum(self.regw.data * conv_norm(samples))

    def reg_w_loss(self):
        eps = torch.randn(*self.weight.size())
        if torch.cuda.is_available():
            eps = eps.cuda()
        mu = self.weight.data
        std = torch.sqrt(self.log_alpha.data.exp() * self.weight.data**2 + 1e-8)
        samples = mu + std * eps
        return torch.sum(self.regw * conv_norm(samples))

    def set_weight_mask(self):
        _, in_filters, height, width = self.weight.size()
        self.mask = self.regw.data < 1.0
        if torch.cuda.is_available():
            self.mask = self.mask.type(torch.cuda.FloatTensor)
        else:
            self.mask = self.mask.type(torch.FloatTensor)
        self.mask = self.mask.unsqueeze(-1).repeat(1, in_filters).unsqueeze(-1).repeat(1, 1, height).unsqueeze(-1).repeat(1, 1, 1, width)
    
    def set_bias_mask(self):
        self.bias_mask = self.regw.data < 1.0
        if torch.cuda.is_available():
            self.bias_mask = self.bias_mask.type(torch.cuda.FloatTensor)
        else:
            self.bias_mask = self.bias_mask.type(torch.FloatTensor)
        return
    
    def apply_mask(self):
        self.weight.data = self.weight.data * self.mask
        if self.bias_mask is not None:
            self.bias.data = self.bias.data * self.bias_mask
        return

    def extra_repr(self):
        return "in_channels={}, out_channels={}, bias={}".format(
            self.in_channels, self.out_channels, self.bias is not None
        )

def test():
    """
    Test Scripts
    """
    dense_layer = TrimDense(3, 4, h=2)
    dense_layer.cuda()

    conv_layer = TrimConv2d(1, 20, kernel_size=(5,5), lamda=0.1, h=9)
    conv_layer.cuda()
    data = torch.randn(100, 1, 28, 28).cuda()

    print (conv_layer(data))
    print (conv_layer.reg_theta_loss())
    return

if __name__ == "__main__":
    test()
