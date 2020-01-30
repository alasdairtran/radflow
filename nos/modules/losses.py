import torch
from torch.nn.modules.loss import _Loss


class SmoothSMAPELoss(_Loss):
    def forward(self, inputs, targets):
        # Get raw view counts
        inputs = torch.exp(inputs) - 1
        targets = torch.exp(targets) - 1

        epsilon = targets.new_tensor(1)
        summ = torch.max(torch.abs(targets) + torch.abs(inputs) + epsilon,
                         0.5 + epsilon)

        numerator = torch.abs(inputs - targets)
        smape = numerator / summ * 2

        return torch.mean(smape)
