import numpy as np





# create synthetic data set
# analogous to original synthetic dataset. See:
# https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network_synth.py#L73

num_points = 501
# ensure num_points is divisible by 3
num_points = 3 * (num_points // 3)
mixture_proportions = [1 / 3.0, 1 / 3.0, 1 / 3.0]
scale = 10.0
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


# set concentration parameter

alpha_0 = 3.


# instantiate classification model
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self,
                 alpha_0=3.):
        super().__init__()
        self.alpha_0 = alpha_0
        self.weights = nn.Linear(
            in_features=2,
            out_features=3)

    def forward(self, x):
        alphas = self.weights(x)
        assert_no_nan_no_inf(alphas)
        output = F.softmax(alphas / self.alpha_0, dim=1)
        assert_no_nan_no_inf(output)
        return output


net = Network(alpha_0=alpha_0)


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


plot_data = [
    go.Scatter(
        x=np.arange(len(losses)),
        y=losses,
        mode='lines')
]

layout = dict(
    title='Negative Log Likelihood Per Batch',
    yaxis=dict(title='NLL'),
    xaxis=dict(title='Batch (size={})'.format(batch_size))
)
fig = go.Figure(data=plot_data, layout=layout)
fig.show()

# define softened labels
one_hot_labels = F.one_hot(labels).to(torch.float32)
epsilon = 0.01
soft_labels = one_hot_labels - one_hot_labels * len(mu) * epsilon + epsilon
soft_concentrations = alpha_0 * soft_labels

# define loss function
# first term of equation 12. See
# https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network_synth.py#L107
from torch.distributions import Dirichlet
from torch.distributions import kl_divergence
from torch.distributions.kl import _kl_dirichlet_dirichlet


def eqn_twelve(model_softmax_outputs, target_concentrations):
    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(model_softmax_outputs)
    kl_divs = _kl_dirichlet_dirichlet(p=target_dirichlet, q=model_dirichlet)
    assert_no_nan_no_inf(kl_divs)
    mean_kl = torch.mean(kl_divs)
    return mean_kl


# create new network
net = Network(alpha_0=alpha_0)
optimizer = optim.SGD(net.parameters(), lr=0.01)

# train!
losses, grad_norms = [], []
for step in range(num_training_steps):
    optimizer.zero_grad()   # zero the gradient buffers
    batch_idx = np.random.choice(
        np.arange(num_points),
        size=batch_size,
        replace=False)
    batch_samples = samples[batch_idx]
    batch_concentrations = soft_concentrations[batch_idx]
    pred_labels = net(batch_samples)
    loss = eqn_twelve(
        model_softmax_outputs=pred_labels,
        target_concentrations=batch_concentrations)
    print('Step: {}, Loss: {}'.format(step, loss.item()))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()


# plot training loss
import plotly.graph_objects as go


plot_data = [
    go.Scatter(
        x=np.arange(len(losses)),
        y=losses,
        mode='lines')
]

layout = dict(
    title='Eqn 12 Term 1 Loss Per Batch',
    yaxis=dict(title='Loss'),
    xaxis=dict(title='Batch (size={})'.format(batch_size))
)
fig = go.Figure(data=plot_data, layout=layout)
fig.show()


# add out of distribution points to our dataset
num_additional_points = 100
num_points += num_additional_points
out_of_dist_samples = torch.tensor(
    np.random.uniform(-100, 100, size=(num_additional_points, 2))).to(torch.float32)
out_of_dist_concentrations = torch.tensor(
    np.ones(shape=(num_additional_points, 3))).to(torch.float32)
samples = torch.cat((samples, out_of_dist_samples))
soft_concentrations = torch.cat((soft_concentrations, out_of_dist_concentrations))



# create new network
net = Network(alpha_0=alpha_0)
optimizer = optim.SGD(net.parameters(), lr=0.01)

# train!
losses, grad_norms = [], []
for step in range(num_training_steps):
    optimizer.zero_grad()   # zero the gradient buffers
    batch_idx = np.random.choice(
        np.arange(num_points),
        size=batch_size,
        replace=False)
    batch_samples = samples[batch_idx]
    batch_concentrations = soft_concentrations[batch_idx]
    pred_labels = net(batch_samples)
    loss = eqn_twelve(
        model_softmax_outputs=pred_labels,
        target_concentrations=batch_concentrations)
    print('Step: {}, Loss: {}'.format(step, loss.item()))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()


# plot training loss
import plotly.graph_objects as go


plot_data = [
    go.Scatter(
        x=np.arange(len(losses)),
        y=losses,
        mode='lines')
]

layout = dict(
    title='Eqn 12 Term 1 Loss Per Batch',
    yaxis=dict(title='Loss'),
    xaxis=dict(title='Batch (size={})'.format(batch_size))
)
fig = go.Figure(data=plot_data, layout=layout)
fig.show()