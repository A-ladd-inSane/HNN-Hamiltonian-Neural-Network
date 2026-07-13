import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class HNN(nn.Module):
    def __init__ (self, batch_size, in_dim = 2, hid_dim = 256, out_dim = 1):
        super(HNN, self).__init__()
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.Softplus(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.Tanh(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.Tanh(),
            nn.Linear(self.hid_dim, self.out_dim),
        )

    def forward(self,x):
        return self.net(x)

    def hamiltonian(self, q, p):
        x = torch.cat([q, p], dim = -1)
        return self. net(x)

    def derivatives (self, q, p):
        x = torch.cat([q, p], dim = -1)
        x.requires_grad_(True)
        H = self.hamiltonian(q, p)
        grads = torch.autograd.grad(H.sum(), x, create_graph = True)[0]
        dH_dq = grads[..., :q.shape[-1]]
        dH_dp = grads[..., q.shape[-1]]
        return dH_dp, -dH_dq

    def loss(self, q, p, dq_dt_target, dp_dt_target):
        dq_dt_pred, dp_dt_pred = self.derivatives(q, p)
        loss_dq_dt = nn.MSELoss()(dq_dt_pred, dq_dt_target)
        loss_dp_dt = nn.MSELoss()(dp_dt_pred, dp_dt_target)
        return (loss_dq_dt + loss_dp_dt), loss_dq_dt, loss_dp_dt