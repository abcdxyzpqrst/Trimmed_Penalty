import torch
import torch.nn.functional as F
import numpy as np
from trim_layers import TrimDense, TrimConv2d
from solvers import fzero, prox_capped_simplex, prox_trimmed_l1
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from utils import get_flat_fts, conv_norm

class TrimMLP(Module):
    def __init__(self, input_dim, num_classes, hidden_size=[300, 100], weight_decay=1,
                 lambdas=[1.0, 1.0, 1.0], alphas=[0, 0, 0], eta=0.001, tau=1.0):
        """
        Args:
            input_dim:      input dimension of data
            num_classes:    number of classes in classification problem
            hidden_size:    hidden layer sizes
            weight_decay:   weight decay rate
            lambdas:        regularization parameters for each layer
            alphas:         belief of sparsity level for each layer
        """
        super(TrimMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.lambdas = lambdas
        self.alphas = alphas
        self.eta = eta
        self.tau = tau
        
        assert len(lambdas) == len(alphas)
        assert len(lambdas) == len(hidden_size) + 1

        # layer construction
        layers = []
        for i, hidden_dim in enumerate(self.hidden_size):
            if i == 0:
                in_dim = self.input_dim
                layers += [TrimDense(in_dim, hidden_dim, lamda=lambdas[i], h=alphas[i]), nn.ReLU()]
            else:
                in_dim = self.hidden_size[i-1]
                layers += [TrimDense(in_dim, hidden_dim, lamda=lambdas[i], h=alphas[i]), nn.ReLU()]
        layers.append(TrimDense(self.hidden_size[-1], num_classes, lamda=lambdas[-1], h=alphas[-1]))
        self.output = nn.Sequential(*layers)

        self.layers = []
        for m in self.modules():
            if isinstance(m, TrimDense):
                self.layers.append(m)
        
        self.reg_w_params = []
        for layer in self.layers:
            self.reg_w_params.append(layer.regw)
        self.reg_w_optimizer = torch.optim.SGD(self.reg_w_params, lr=self.tau)

    def forward(self, x):
        return self.output(x)
    
    def kld(self):
        kld = 0.0
        for layer in self.layers:
            kld += layer.kld()
        return kld
    
    def reg_w_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            reg_loss += layer.reg_w_loss()
        return reg_loss

    def reg_theta_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            reg_loss += layer.reg_theta_loss()
        return reg_loss

    def trimmed_l1_regularization(self):
        # gradient over w using sampling
        reg_w = self.reg_w_loss()
        self.reg_w_optimizer.zero_grad()
        reg_w.backward()
        self.reg_w_optimizer.step()
        
        # projection onto the capped simplex
        for layer in self.layers:
            prox_capped_simplex(layer.regw, 0.0, 1.0, layer.n_pnt) 
        return
    
    def trimming_loss(self):
        """
        For testing regularization loss
        Therefore, we do not use sampling method here.
        """
        reg_loss = 0.0
        for layer in self.layers:
            reg_loss += torch.sum(layer.regw.data * torch.norm(layer.weight.data, dim=1)).item()
        return reg_loss
   
    def set_mask(self, bias=False):
        if not bias:
            for layer in self.layers:
                layer.set_weight_mask()
        else:
            for i, layer in enumerate(self.layers):
                layer.set_weight_mask()
                if i == 0:
                    continue
                else:
                    self.layers[i-1].set_bias_mask(layer.regw)
        return

    def apply_mask(self):
        for layer in self.layers:
            layer.apply_mask()
        return
    
    def get_expected_flops(self):
        """
        To be implemented
        """
        return
    
    def architecture(self):
        s = ""
        for layer in self.layers:
            neurons = (layer.regw < 1.0).sum().item()
            s += str(neurons)
            s += "-"
        s += "10"
        return s

class TrimLeNet5(Module):
    def __init__(self, input_size, num_classes, conv_dims=[20, 50], fc_dims=500, 
                 lambdas=[1.0, 1.0, 1.0, 1.0], alphas=[0, 0, 0, 0], eta=0.001, tau=0.001):
        super(TrimLeNet5, self).__init__()
        self.input_size = input_size
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.lambdas = lambdas
        self.alphas = alphas
        self.eta = eta
        self.tau = tau
        
        assert len(self.lambdas) == len(self.alphas)
        assert len(self.lambdas) == len(conv_dims) + 2

        conv_layers = [TrimConv2d(input_size[0], conv_dims[0], kernel_size=(5,5), 
                                  lamda=lambdas[0], h=alphas[0]), 
                       nn.ReLU(), nn.MaxPool2d(2),
                       TrimConv2d(conv_dims[0], conv_dims[1], kernel_size=(5,5),
                                  lamda=lambdas[1], h=alphas[1]), 
                       nn.ReLU(), nn.MaxPool2d(2)]
        self.conv_layers = nn.Sequential(*conv_layers)

        flat_fts = get_flat_fts(input_size, self.conv_layers)
        
        dense_layers = [TrimDense(flat_fts, self.fc_dims, lamda=lambdas[2], h=alphas[2]),
                        nn.ReLU(),
                        TrimDense(self.fc_dims, num_classes, lamda=lambdas[-1], h=alphas[-1])]
        self.dense_layers = nn.Sequential(*dense_layers)

        self.layers = []
        for m in self.modules():
            if isinstance(m, TrimDense) or isinstance(m, TrimConv2d):
                self.layers.append(m)

        self.reg_w_params = []
        for layer in self.layers:
            self.reg_w_params.append(layer.regw)
        self.reg_w_optimizer = torch.optim.SGD(self.reg_w_params, lr=self.tau)

    def forward(self, x):
        o = self.conv_layers(x)
        o = o.view(o.size(0), -1)
        return self.dense_layers(o)

    def kld(self):
        kld = 0.0
        for layer in self.layers:
            kld += layer.kld()
        return kld

    def reg_theta_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            reg_loss += layer.reg_theta_loss()
        return reg_loss

    def reg_w_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            reg_loss += layer.reg_w_loss()
        return reg_loss

    def trimmed_l1_regularization(self):
        reg_w = self.reg_w_loss()
        self.reg_w_optimizer.zero_grad()
        reg_w.backward()
        self.reg_w_optimizer.step()
        
        for layer in self.layers:
            prox_capped_simplex(layer.regw, 0.0, 1.0, layer.n_pnt)
        return
    
    def trimming_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, TrimConv2d):
                reg_loss += torch.sum(layer.regw.data * conv_norm(layer.weight.data)).item()
            else:
                reg_loss += torch.sum(layer.regw.data * torch.norm(layer.weight.data, dim=1)).item()
        return reg_loss
    
    def set_mask(self, bias=False):
        if not bias:
            for layer in self.layers:
                layer.set_weight_mask() 
        else:
            for i, layer in enumerate(self.layers):
                layer.set_weight_mask()
                if isinstance(layer, TrimConv2d):
                    layer.set_bias_mask()
                else:
                    if i == 2:
                        continue
                    else:
                        self.layers[i-1].set_bias_mask(layer.regw)

    def apply_mask(self):
        for layer in self.layers:
            layer.apply_mask()
        return
    
    def architecture(self):
        s = ""
        for layer in self.layers:
            features = (layer.regw < 1.0).sum().item()
            s += str(features)
            s += "-"
        s += "10"
        return s

def test():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    model = TrimLeNet5(input_size=(1, 28, 28), num_classes=10, lambdas=[0.1, 0.1, 0.1, 0.1], alphas=[9, 18, 65, 25])
    model.cuda()

    print (model.reg_theta_loss())
    print (model.reg_w_loss())
    print (model.architecture())
    
    model.set_mask()
    for layer in model.layers:
        print (layer.mask.size())
    
    print (model.trimming_loss())
    return

if __name__ == "__main__":
    test() 
