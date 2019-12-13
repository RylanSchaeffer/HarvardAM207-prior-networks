import torch
import torch.nn as nn
import torch.nn.functional as F


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


class OneLayer(torch.nn.Module):

    def __init__(self,
                 n_in,
                 n_out,
                 alpha_0=1.):

        super(OneLayer, self).__init__()
        self.alpha_0 = alpha_0
        self.feedforward_layers = nn.Sequential(
            nn.Linear(n_in, n_out))

    def forward(self, x):
        logits = self.feedforward_layers(x)
        # TODO: ask Theo what means is
        means = F.softmax(logits / self.alpha_0, dim=1)  # I have doubts about alpha 0..
        alphas = torch.exp(logits)
        precision = torch.sum(alphas)
        return logits, means, alphas, precision


class TwoLayer(nn.Module):

    def __init__(self,
                 n_in,
                 n_out,
                 alpha_0=1.,
                 n_hidden=12):

        super(TwoLayer, self).__init__()
        self.alpha_0 = alpha_0
        self.feedforward_layers = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_out))

    def forward(self, x):
        logits = self.feedforward_layers(x)
        # TODO: ask Theo what mean is
        alphas = torch.exp(logits) + 1
        mean = alphas / alphas.sum(dim=1).unsqueeze(dim=1)
        precision = torch.sum(alphas)
        return logits, mean, alphas, precision