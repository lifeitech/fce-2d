"""
Implementation of Glow[1] model in PyTorch for 2D dataset. Adapted from
https://github.com/kamenbliznashki/normalizing_flows/blob/master/glow.py

[1] https://arxiv.org/abs/1807.03039
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# --------------------
# Model component layers
# --------------------
class Actnorm(nn.Module):
    def __init__(self, param_dim=(1,2)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        if not self.initialized:
            # x.shape = (B, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, unbiased=False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = - self.scale.abs().log().sum() 
        return z, logdet

    def inverse(self, z):
        x = z * self.scale + self.bias
        logdet = self.scale.abs().log().sum()
        return x, logdet


class Invertible1x1Conv(nn.Module):
    def __init__(self, dim=2):
        super().__init__()

        w = torch.randn(dim, dim)
        w = torch.linalg.qr(w)[0]   # W^{-1} = W^T (only at initialization)
        self.w = nn.Parameter(w)

    def forward(self, x):        
        logdet = torch.slogdet(self.w)[-1]
        return x @ self.w.t(), logdet  # (WX)^T = X^TW^T = y^T.

    def inverse(self, z):
        w_inv = self.w.t().inverse()
        logdet = - torch.slogdet(self.w)[-1]
        return z @ w_inv, logdet


class AffineCoupling(nn.Module):
    def __init__(self, dim=2, width=128):
        super().__init__()
        self.fc1 = nn.Linear(dim//2, width, bias=True) 
        self.actnorm1 = Actnorm(param_dim=(1, width))
        self.fc2 = nn.Linear(width, width, bias=True)
        self.actnorm2 = Actnorm(param_dim=(1, width))
        self.fc3 = nn.Linear(width, dim, bias=True)
        self.log_scale_factor = nn.Parameter(torch.zeros(1,2))
        
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()

    def forward(self, x):
        x_a, x_b = x.chunk(2, dim=1)  # x.shape = [batch, 2]

        h = F.relu(self.actnorm1(self.fc1(x_b))[0])
        h = F.relu(self.actnorm2(self.fc2(h))[0])
        h = self.fc3(h) * self.log_scale_factor.exp()
        t = h[:,0::2]  # shift; take even dimension(s)
        s = h[:,1::2]  # scale; take odd dimension(s)
        s = torch.sigmoid(s + 2.)

        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)  # z.shape = [batch, 2]
        logdet = s.log().sum(1)

        return z, logdet

    def inverse(self, z):
        z_a, z_b = z.chunk(2, dim=1)  

        h = F.relu(self.actnorm1(self.fc1(z_b))[0])
        h = F.relu(self.actnorm2(self.fc2(h))[0])
        h = self.fc3(h)  * self.log_scale_factor.exp()
        t = h[:,0::2]  # shift; take even dimension(s)
        s = h[:,1::2]  # scale; take odd dimension(s)
        s = torch.sigmoid(s + 2.)

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1) 

        logdet = - s.log().sum(1)
        return x, logdet

# --------------------
# Container layers
# --------------------
class FlowSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        slogdet = 0.
        for module in self:
            x, logdet = module(x)
            slogdet = slogdet + logdet
        return x, slogdet

    def inverse(self, z):
        slogdet = 0.
        for module in reversed(self):
            z, logdet = module.inverse(z)
            slogdet = slogdet + logdet
        return z, slogdet


class FlowStep(FlowSequential):
    """ One step (Actnorm -> Invertible 1x1 conv -> Affine coupling) """
    def __init__(self, dim=2, width=128):
        super().__init__(
                        Actnorm(param_dim=(1,dim)),
                        Invertible1x1Conv(dim=dim),
                        AffineCoupling(dim=dim, width=width))


class FlowLevel(nn.Module):
    """ One depth (e.g. 10) (FlowStep x 10) """
    def __init__(self, dim=2, width=128, depth=10):
        super().__init__()
        self.flowsteps = FlowSequential(*[FlowStep(dim, width) for _ in range(depth)])  # original: FlowStep(4*n_channels, width)

    def forward(self, x):
        z, logdet = self.flowsteps(x)
        return z, logdet

    def inverse(self, z):
        x, logdet = self.flowsteps.inverse(z)
        return x, logdet


# --------------------
# Model
# --------------------
class Glow(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L"""
    def __init__(self, width=128, depth=10, n_levels=1, data_dim=2):
        super().__init__()

        # (FlowStep x depth) x n_levels
        self.flowlevels = nn.ModuleList([FlowLevel(dim=data_dim, width=width, depth=depth) for i in range(n_levels)])
        self.flowstep = FlowSequential(*[FlowStep(dim=data_dim, width=width) for _ in range(depth)])

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(2))
        self.register_buffer('base_dist_var', torch.eye(2))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x):
        slogdet = 0
        for m in self.flowlevels:
            z, logdet = m(x)
            slogdet = slogdet + logdet
        z, logdet = self.flowstep(z)
        slogdet = slogdet + logdet
        return z, slogdet

    def inverse(self, z=None, batch_size=32, z_std=1.):
        if z is None:
            z = z_std * self.base_dist.sample((batch_size,))
        x, slogdet = self.flowstep.inverse(z)
        for m in reversed(self.flowlevels):
            x, logdet = m.inverse(x)
            slogdet = slogdet + logdet

        # get logq(x̃), where x̃ = f^{-1}(z)
        logq_gen =  (self.base_dist.log_prob(z) - slogdet).unsqueeze(1)

        return x, logq_gen

    def log_prob(self, x):
        z, logdet = self.forward(x)
        log_prob = self.base_dist.log_prob(z) + logdet
        return log_prob.unsqueeze(1)