import argparse
import logging
import numpy as np
import plotly.graph_objects as go
import torch
from torch.distributions import Dirichlet
from torch.distributions.kl import _kl_dirichlet_dirichlet
import torch.nn.functional as F
import tqdm

import models


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def create_args():
    parser = create_arg_parser()
    args = parser.parse_args()
    return args


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse',
                        help='Use Backward KL. If unspecified, use Forward KL',
                        action='store_true')
    parser.add_argument('--n_epochs',
                        help='Number of epochs to train',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='Number of data per batch',
                        type=int,
                        default=32)
    parser.add_argument('--scale',
                        help="Scale of the synt. data",
                        type=float,
                        default=10.0)
    parser.add_argument('--lr',
                        help="learning rate",
                        type=float,
                        default=0.01)
    return parser


def create_data(args,
                points_per_cluster=100,
                n_clusters=3,
                scale=10,
                print=True):
    """
    #TODO-implement so that we can actually choose other inputs
    - Args to specify the mean variances if need be, under the form mu_1, sigma_1" and so on
    output: np.array with samples from each gaussian.
    """
    means = np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.],
        [-20., -20.]])
    means *= 2

    covariances = np.array([
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
    ])

    n_train_samples_per_gaussian = np.array([100, 100, 100, 100])

    out_of_distribution_gaussians = np.array([False, False, False, True])

    mog_samples, mog_labels, mog_concentrations = create_data_mixture_of_gaussians(
        means=means,
        covariances=covariances,
        n_samples_per_gaussian=n_train_samples_per_gaussian,
        out_of_distribution_gaussians=out_of_distribution_gaussians)

    x_train = torch.tensor(
        mog_samples,
        dtype=torch.float)
    labels_train = torch.tensor(
        mog_labels,
        dtype=torch.long)
    concentrations_train = torch.tensor(
        mog_concentrations,
        dtype=torch.float32)

    assert_no_nan_no_inf(x_train)
    assert_no_nan_no_inf(labels_train)
    assert_no_nan_no_inf(concentrations_train)

    data = dict(
        x_train=x_train,
        labels_train=labels_train,
        concentrations_train=concentrations_train,
        x_test=x_train,
        labels_test=labels_train,
        concentrations_test=concentrations_train)

    return data


def create_data_mixture_of_gaussians(means,
                                     covariances,
                                     n_samples_per_gaussian,
                                     out_of_distribution_gaussians,
                                     prev_samples=None,
                                     prev_labels=None,
                                     prev_concentrations=None):
    """
    Creates a mixture of Gaussians with corresponding labels and Dirichlet
    concentration parameters. If x_data and y_data are given, append the new
    samples to x_data, y_data and return those instead.


    :param prev_samples:
    :param prev_concentrations:
    :param prev_labels:
    :param means: shape (number of gaussians, dim of gaussian)
    :param covariances: shape (number of gaussians, dim of gaussian, dim_of_gaussian)
    :param n_samples_per_gaussian: int, shape (number of gaussians, )
    :param out_of_distribution_gaussians: bool,  shape (number of gaussians, )
    :return mog_samples: shape (sum(n_samples_per_gaussian), dim of gaussian)
    :return mog_labels: int, shape (sum(n_samples_per_gaussian), )
    :return mog_concentrations: shape (sum(n_samples_per_gaussian), dim of gaussian)
    """

    dim_of_gaussians = means.shape[1]
    n_total_samples = n_samples_per_gaussian.sum()

    # preallocate arrays to store new samples
    mog_samples = np.zeros(shape=(n_total_samples, dim_of_gaussians))
    mog_labels = np.zeros(shape=(n_total_samples,), dtype=np.int)

    write_index = 0
    for i, (mean, covariance, n_samples) in \
            enumerate(zip(means, covariances, n_samples_per_gaussian)):
        # generate and save samples
        gaussian_samples = np.random.multivariate_normal(
            mean=mean, cov=covariance, size=n_samples)
        mog_samples[write_index:write_index + n_samples] = gaussian_samples

        # generate and save labels
        gaussian_labels = np.full(shape=n_samples, fill_value=i)
        mog_labels[write_index:write_index + n_samples] = gaussian_labels

        write_index += n_samples

    # generate concentrations
    n_gaussians = (~out_of_distribution_gaussians).sum()
    mog_concentrations = np.ones((n_total_samples, n_gaussians))
    in_distribution_rows = np.isin(
        mog_labels,
        np.argwhere(~out_of_distribution_gaussians))
    mog_concentrations[in_distribution_rows, mog_labels[in_distribution_rows]] += 100

    # shuffle data
    shuffle_indices = np.random.choice(
        np.arange(n_total_samples),
        size=n_total_samples,
        replace=False)
    mog_samples = mog_samples[shuffle_indices]
    mog_labels = mog_labels[shuffle_indices]
    mog_concentrations = mog_concentrations[shuffle_indices]

    # append new samples to previous data, if provided
    if prev_samples is not None:
        # check that other arrays are also not None
        assert prev_labels is not None
        assert prev_concentrations is not None

        mog_samples = np.concatenate((mog_samples, prev_samples), axis=0)
        mog_labels = np.concatenate((mog_labels, prev_labels), axis=0)
        mog_concentrations = np.concatenate((mog_concentrations, prev_concentrations), axis=0)

    return mog_samples, mog_labels, mog_concentrations


def create_loss_fn():
    return kl_loss
    # return torch.nn.CrossEntropyLoss


def create_model():
    model = models.TwoLayer(n_in=2, n_out=3, n_hidden=12)
    return model


def create_optimizer(model, lr=0.01):
    for name, param in model.named_parameters():
        print(name, "Requires grad?", param.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return optimizer


def eval_model(model,
               x_test,
               y_test):
    # set model in eval mode
    model.eval()

    # TODO-Add a sequential test loader
    y_hat, means, alphas, precision = model(x_test)
    assert_no_nan_no_inf(y_hat)
    pred_proba, pred_class = torch.max(means, 1)
    _, true_class = torch.max(y_test, 1)
    accuracy = torch.div(
        (pred_class == true_class).sum().type(torch.float32),
        len(y_test))
    return accuracy, pred_proba, pred_class


def plot_all(x_train,
             labels_train,
             concentrations_train,
             model,
             training_loss):
    plot_training_data(x_train=x_train, labels_train=labels_train)
    plot_training_loss(training_loss=training_loss)
    plot_decision_surface(model=model, x_train=x_train, labels_train=labels_train)


def plot_training_data(x_train,
                       labels_train):
    plot_data = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        mode='markers',
        marker=dict(color=labels_train))

    layout = dict(
        title='Training Data',
        xaxis=dict(title='x'),
        yaxis=dict(title='y'))

    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()


def plot_decision_surface(model,
                          x_train,
                          labels_train):
    
    # discretize input space i.e. all possible pairs of coordinates
    # between [-40, 40] x [-40, 40]
    possible_vals = np.linspace(-40, 40, 81)
    x_vals, y_vals = np.meshgrid(possible_vals, possible_vals)
    grid_inputs = np.stack((x_vals.flatten(), y_vals.flatten()), axis=1)
    grid_inputs = torch.tensor(grid_inputs, dtype=torch.float32)

    # forward pass model
    y_hat, means, alphas, precision = model(grid_inputs)

    strength_of_mean = torch.max(means, dim=1)[0].detach().numpy()

    plot_data = [
        # add model outputs
        go.Surface(x=possible_vals,
                   y=possible_vals,
                   z=strength_of_mean.reshape(x_vals.shape)),
        # add training points
        go.Scatter3d(x=x_train[:, 0],
                     y=x_train[:, 1],
                     z=1.1 * np.ones(x_train.shape[0]),
                     mode='markers',
                     marker=dict(color=labels_train))
    ]

    # TODO: Label axes

    fig = go.Figure(data=plot_data)
    fig.show()


def plot_training_loss(training_loss):
    plot_data = go.Scatter(
        x=np.arange(len(training_loss)),
        y=training_loss)

    layout = dict(
        title='Training Loss per Gradient Step',
        xaxis=dict(title='Gradient Step'),
        yaxis=dict(title='Training Loss'))

    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()


def setup(args):
    model = create_model()
    optimizer = create_optimizer(model=model)
    loss_fn = create_loss_fn()
    data = create_data(args=args)
    device = "gpu:0" if torch.cuda.is_available() else "cpu"

    # TODO: has anyone checked that the following line actually places
    # tensors on the GPU? 
    device = torch.device(device)
    logging.info('Working on device: ', device)

    return model, optimizer, loss_fn, data


def train_model(model,
                optimizer,
                loss_fn,
                n_epochs,
                batch_size,
                x_train,
                target_concentrations):

    # TODO: Keep track of the model_concentrations parameters. Maybe an output dictionnary so that we can
    # keep track or more/less things that we want to investigate, e.g:
    # tracks = {'training_loss':[], 'model_concentrations':[], ...}

    # set model in training mode
    model.train()

    # train model
    training_loss = []
    num_samples = x_train.shape[0]
    for epoch in range(n_epochs):
        for _ in range(num_samples // batch_size):

            # randomly sample indices for batch
            batch_indices = np.random.choice(
                np.arange(x_train.shape[0]),
                size=batch_size,
                replace=False)
            x_train_batch = x_train[batch_indices]
            target_concentrations_batch = target_concentrations[batch_indices]

            optimizer.zero_grad()
            logits, mean, model_concentrations, precision = model(x_train_batch)
            assert_no_nan_no_inf(model_concentrations)
            batch_loss = loss_fn(model_concentrations, target_concentrations_batch)
            assert_no_nan_no_inf(batch_loss)
            training_loss.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()
        print("Last obtained batch_loss", batch_loss.item())

    return model, optimizer, training_loss


def get_target_dirichlet(y_train, alpha_0, nb_class=3, epsilon=1e-4):
    """Input: 
        - alpha_0 : Hyperparameter to specify the sharpness.
        - epsilon: Smoothing parameter
        - nb_class: Explicit
    Output: target_dirichlet: torch.distributions.Dirichlet object
        access to concentration parameters with output.concentration"""
    one_hot_labels = F.one_hot(y_train.squeeze()).to(torch.float32)
    soft_labels = one_hot_labels - one_hot_labels * nb_class * epsilon + epsilon
    target_concentrations = alpha_0 * soft_labels
    # target_dirichlet = Dirichlet(target_concentrations)
    return target_concentrations


def kl_loss(model_concentrations, target_concentrations, reverse=True):
    """
    Input: Model concentrations, target concentrations parameters.
    Output: Average of the KL between the two Dirichlet.
    """
    assert torch.all(model_concentrations > 0)
    assert torch.all(target_concentrations > 0)

    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(model_concentrations)
    if reverse:
        kl_divs = _kl_dirichlet_dirichlet(
            p=model_dirichlet,
            q=target_dirichlet)
    else:
        kl_divs = _kl_dirichlet_dirichlet(
            p=target_dirichlet,
            q=model_dirichlet)
    assert_no_nan_no_inf(kl_divs)
    mean_kl = torch.mean(kl_divs)
    return mean_kl
