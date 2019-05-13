import math
import torch
import numpy as np 

def fzero(f, a, b, xtol=2e-12, rtol=8e-16):
    """
    finding roots of a function by bisection method
    this function should be loaded on **CPU**
    Args:
        f:  function
        a:  left endpoint
        b:  right endpoint
        xtol, rtol --> tolerance
    """
    if f(a)*f(b) > 0:
        print ("The function has positive value at the endpoints of interval")
        print ("The root cannot be found using bisection method, use some other method")
        return

    elif f(a) == 0:
        return a
    
    elif f(b) == 0:
        return b

    else:
        dist = b - a        # distance
        while True:
            dist *= 0.5
            m = a + dist
            if f(m)*f(a) >= 0:
                a = m 
            if f(m) == 0 or abs(dist) < xtol + rtol*abs(m):
                return m
        print ("if you are here, error!")
        return

def prox_capped_simplex(w, lb, ub, h):
    """
    Projection mapping onto the capped simplex
    This function should be loaded on **CPU**
    Args:
        w (torch.tensor):   the weight on parameters
        lb (scalar):        lower bound
        ub (scalar):        upper bound
        h (scalar):         the trimming parameters
    """
    total = len(w)
    if h == total:
        w.data.fill_(1.0)
        return
    
    elif h > total:
        print ("error: wrong h or size of vectors!")
        return

    else:
        h = float(h)
        def f(alpha):
            return (w - alpha).data.clamp(min=0.0, max=1.0).sum().item() - h
        m = -1.5 + w.min().item()           # scalar
        M = w.max().item()                  # scalar
        r = fzero(f, m, M)
        
        w.data.sub_(r)
        w.data.clamp_(min=lb, max=ub)
        return

def prox_l1(theta, kappa):
    out_features = theta.data.size(1)
    scale = 1 - torch.div(kappa, torch.norm(theta.data, dim=1))
    scale.data.clamp_(min=0.0)
    theta.data = torch.mul(scale.unsqueeze(-1).repeat(1, out_features), theta.data)
    return

def prox_trimmed_l1(theta, w, kappa):
    """
    Proximal trimmed group \ell_1 with \ell_2 grouping
    Args:
        theta (torch.nn.Parameter): parameters
        w (torch.tensor):           reg. weight on parameters
        kappa (scalar):             soft-thresholding (which is eta * lambda)
    """
    out_features = theta.data.size(1)
    scale = 1 - torch.div(kappa * w.data, torch.norm(theta.data, dim=1))
    scale.data.clamp_(min=0.0)
    theta.data = torch.mul(scale.unsqueeze(-1).repeat(1, out_features), theta.data)
    return

def test():
    """
    Test scripts for implemented functions
    """
    hidden_size = 784
    sparse_list = [i for i in range(100)]
    device = 'cuda'
    for sparsity_level in sparse_list:
        h = sparsity_level
        w = np.ones(hidden_size) * (h/hidden_size)
        lb = 0.0
        ub = 1.0
        def f(alpha):
            lbs = np.ones(len(w)) * lb
            ubs = np.ones(len(w)) * ub
            return np.maximum(lbs, np.minimum(ubs, w - alpha)).sum() - h 
        m = -1.5 + w.min()
        M = w.max()
        r = fzero(f, m, M)
        print ("True Sparsity: {}, Estm Sparsity: {}".format(sparsity_level, round(f(r) + h)))
    return

if __name__ == "__main__":
    test()
