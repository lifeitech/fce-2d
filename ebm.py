import torch
import torch.nn as nn


class EBM(nn.Module):
    def __init__(self):
        super(EBM, self).__init__()
        # The normalizing constant logZ(θ)        
        self.c = nn.Parameter(torch.tensor([1.], requires_grad=True))

        self.f = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            )

    def forward(self, x):
        log_prob = - self.f(x) - self.c
        return log_prob