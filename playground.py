import numpy as np


# create synthetic data set
# analogous to original synthetic dataset. See:
# https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network_synth.py#L73

num_points = 501
# ensure num_points is divisible by 3
num_points = 3 * (num_points // 3)
mixture_proportions = [1 / 3.0, 1 / 3.0, 1 / 3.0]
scale = 4.0
mu = [
    scale * np.asarray([0, 1.0], dtype=np.float32),
    scale * np.asarray([-np.sqrt(3) / 2, -1. / 2], dtype=np.float32),
    scale * np.asarray([np.sqrt(3) / 2, -1. / 2], dtype=np.float32)
]
diag_cov = 2 * np.identity(2)

samples, labels = [], []
for i in range(len(mu)):
    num_class_samples = int(num_points*mixture_proportions[i])
    class_samples = np.random.multivariate_normal(
        mean=mu[i],
        cov=diag_cov,
        size=num_class_samples)
    samples.append(class_samples)
    class_labels = np.full(
        shape=num_class_samples,
        fill_value=i)
    labels.append(class_labels)

samples = np.concatenate(samples)
labels = np.concatenate(labels)

# shuffle dataset
shuffle_idx = np.random.choice(
    np.arange(num_points),
    size=num_points,
    replace=False)
samples = samples[shuffle_idx, :]
labels = labels[shuffle_idx]


# plot dataset
import plotly.express as px


fig = px.scatter(
    x=samples[:, 0],
    y=samples[:, 1],
    color=labels,
    labels=labels)
# fig.show()



# instantiate classification model
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Linear(
            in_features=2,
            out_features=3)

    def forward(self, x):
        output = F.softmax(self.weights(x), dim=1)
        return output


net = Network()


# define typical classification function
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()


# train model
import torch.optim as optim


optimizer = optim.SGD(net.parameters(), lr=0.01)
num_training_steps = 100
batch_size = 20

# convert data to tensors
samples = torch.Tensor(samples)
# labels = F.one_hot(torch.Tensor(labels).to(torch.int64), num_classes=len(mu))
labels = torch.Tensor(labels).to(torch.int64)

# train!
losses = []
for step in range(num_training_steps):
    optimizer.zero_grad()   # zero the gradient buffers
    batch_idx = np.random.choice(
        np.arange(num_points),
        size=batch_size,
        replace=False)
    batch_samples, batch_labels = samples[batch_idx], labels[batch_idx]
    pred_labels = net(batch_samples)
    loss = criterion(pred_labels, batch_labels)
    print('Step: {}, Loss: {}'.format(step, loss.item()))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()


# plot training loss
import plotly.graph_objects as go


fig = go.Figure(go.Scatter(
    x=np.arange(len(losses)),
    y=losses,
    mode='lines'))
fig.show()

# define softened labels
one_hot_labels = F.one_hot(labels).to(torch.float32)
epsilon = 0.01
soft_labels = one_hot_labels - one_hot_labels*len(mu)*epsilon + epsilon


# define loss function
# first term of equation 12. See
# https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network_synth.py#L107
from torch.distributions import Dirichlet
from torch.distributions.kl import kl_divergence


def eqn_twelve(model_softmax_outputs, soft_labels):
    target_dirichlet = Dirichlet(soft_labels)
    model_dirichlet = Dirichlet(model_softmax_outputs)
    kl_div = kl_divergence(target_dirichlet, model_dirichlet)
    return torch.mean(kl_div)


# train!
losses = []
for step in range(num_training_steps):
    optimizer.zero_grad()   # zero the gradient buffers
    batch_idx = np.random.choice(
        np.arange(num_points),
        size=batch_size,
        replace=False)
    batch_samples, batch_labels = samples[batch_idx], soft_labels[batch_idx]
    pred_labels = net(batch_samples)
    loss = eqn_twelve(
        model_softmax_outputs=pred_labels,
        soft_labels=batch_labels)
    print('Step: {}, Loss: {}'.format(step, loss.item()))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()


# plot training loss
import plotly.graph_objects as go


fig = go.Figure(go.Scatter(
    x=np.arange(len(losses)),
    y=losses,
    mode='lines'))
fig.show()