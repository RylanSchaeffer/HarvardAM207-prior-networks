import argparse
import logging
import numpy as np
import plotly.graph_objects as go
import torch
from torch.distributions import Categorical, Dirichlet
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
                        default=500)
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


def create_loss_fn(reverse):
    if reverse:
        return kl_backward
    else:
        return kl_forward


def create_model(in_dim,
                 out_dim):
    model = models.TwoLayer(n_in=in_dim, n_out=out_dim, n_hidden=12)
    return model


def create_optimizer(model, lr=0.01):
    for name, param in model.named_parameters():
        print(name, "Requires grad?", param.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return optimizer


def entropy_categorical(categorical_parameters):
    entropy = Categorical(categorical_parameters).entropy()
    entropy = entropy.detach().numpy()
    return entropy


def entropy_dirichlet(dirichlet_concentrations):
    entropy = Dirichlet(dirichlet_concentrations).entropy()
    entropy = entropy.detach().numpy()
    return entropy


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


def kl_backward(model_concentrations, target_concentrations):
    """
    Input: Model concentrations, target concentrations parameters.
    Output: Average of the KL between the two Dirichlet.
    """
    assert torch.all(model_concentrations > 0)
    assert torch.all(target_concentrations > 0)

    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(model_concentrations)
    kl_divs = _kl_dirichlet_dirichlet(
        p=model_dirichlet,
        q=target_dirichlet)
    assert_no_nan_no_inf(kl_divs)
    mean_kl = torch.mean(kl_divs)
    return mean_kl


def kl_forward(model_concentrations, target_concentrations):
    """
    Input: Model concentrations, target concentrations parameters.
    Output: Average of the KL between the two Dirichlet.
    """
    assert torch.all(model_concentrations > 0)
    assert torch.all(target_concentrations > 0)

    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(model_concentrations)
    kl_divs = _kl_dirichlet_dirichlet(
        p=target_dirichlet,
        q=model_dirichlet)
    assert_no_nan_no_inf(kl_divs)
    mean_kl = torch.mean(kl_divs)
    return mean_kl


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

    entropy_of_means = entropy_categorical(categorical_parameters=means)

    plot_data = [
        # add model outputs
        go.Surface(x=possible_vals,
                   y=possible_vals,
                   z=entropy_of_means.reshape(x_vals.shape)),
        # add training points
        go.Scatter3d(x=x_train[:, 0],
                     y=x_train[:, 1],
                     z=1.1 * np.full(x_train.shape[0], fill_value=np.max(entropy_of_means)),
                     mode='markers',
                     marker=dict(color=labels_train))
    ]

    layout = dict(
        title='Decision Surface',
        scene=dict(
            zaxis=dict(title='Entropy of Mu'),
            xaxis=dict(title='input_dim_1'),
            yaxis=dict(title='input_dim_2')
        )
    )

    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()


def plot_results(train_samples,
                 labels_train,
                 train_concentrations,
                 model,
                 training_loss):

    plot_training_data(x_train=train_samples, labels_train=labels_train)
    plot_training_loss(training_loss=training_loss)
    plot_decision_surface(model=model, x_train=train_samples, labels_train=labels_train)


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


def setup(args,
          in_dim,
          out_dim):
    model = create_model(
        in_dim=in_dim,
        out_dim=out_dim)
    optimizer = create_optimizer(model=model)
    loss_fn = create_loss_fn(reverse=args.reverse)
    device = "gpu:0" if torch.cuda.is_available() else "cpu"

    # TODO: check that the following code places tensors on the GPU
    device = torch.device(device)
    logging.info('Working on device: ', device)

    return model, optimizer, loss_fn


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
