import torch
import torch.nn as nn
import torch.nn.functional as F


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


class FeedforwardNetwork(torch.nn.Module):

    def __init__(self,
                 n_in,
                 n_out,
                 n_per_hidden_layer=None,
                 alpha_0=1.):

        super(FeedforwardNetwork, self).__init__()
        self.alpha_0 = alpha_0
        self.n_in = n_in
        self.n_out = n_out
        self.n_per_hidden_layer = n_per_hidden_layer
        self.feedforward_layers = self.create_feedforward_layers()

    def create_feedforward_layers(self):

        layers = []
        n_prev = self.n_in
        for n_hidden in self.n_per_hidden_layer:
            layers.append(nn.Linear(in_features=n_prev, out_features=n_hidden))
            layers.append(nn.LeakyReLU())
            n_prev = n_hidden
        layers.append(nn.Linear(in_features=n_prev, out_features=self.n_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        logits = self.feedforward_layers(x)
        assert_no_nan_no_inf(logits)

        concentrations = torch.exp(logits) + 1
        assert_no_nan_no_inf(concentrations)

        mean = concentrations / concentrations.sum(dim=1).unsqueeze(dim=1)
        assert_no_nan_no_inf(mean)

        precision = torch.sum(concentrations)
        assert_no_nan_no_inf(precision)

        y_pred = F.softmax(concentrations / self.alpha_0, dim=1)
        assert_no_nan_no_inf(y_pred)

        model_outputs = {
            'logits': logits,
            'mean': mean,
            'concentrations': concentrations,
            'precision': precision,
            'y_pred': y_pred
        }
        return model_outputs


class LogisticRegression(torch.nn.Module):

    def __init__(self,
                 n_in,
                 n_out,
                 alpha_0=3.,
                 ):
        super().__init__()
        self.alpha_0 = alpha_0
        self.weights = nn.Linear(
            in_features=n_in,
            out_features=n_out)

    def forward(self, x):
        alphas = self.weights(x)
        assert_no_nan_no_inf(alphas)
        logits = alphas / self.alpha_0
        y_pred = F.softmax(alphas / self.alpha_0, dim=1)
        assert_no_nan_no_inf(y_pred)
        model_outputs = {
            'logits': logits,
            'alphas': alphas,
            'y_pred': y_pred
        }
        return model_outputs