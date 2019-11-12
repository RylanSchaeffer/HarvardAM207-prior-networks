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
    mu_1, sigma_1 = scale * \
                    np.array([0.0, 1.0]), np.array([[2.0, 0], [0, 2.0]])
    mu_2, sigma_2 = scale * \
                    np.array([-np.sqrt(3) / 2, -1. / 2]), np.array([[2.0, 0], [0, 2.0]])
    mu_3, sigma_3 = scale * \
                    np.array([np.sqrt(3) / 2, -1. / 2]), np.array([[2.0, 0], [0, 2.0]])

    X_1 = np.random.multivariate_normal(
        mean=mu_1, cov=sigma_1, size=points_per_cluster)
    X_2 = np.random.multivariate_normal(
        mean=mu_2, cov=sigma_2, size=points_per_cluster)
    X_3 = np.random.multivariate_normal(
        mean=mu_3, cov=sigma_3, size=points_per_cluster)

    Y_1, Y_2, Y_3 = np.zeros((len(X_1), 1)), np.ones(
        (len(X_1), 1)), 2 * np.ones((len(X_1), 1))
    # if print:
    #     fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    #     ax.scatter(X_1[:, 0], X_1[:, 1], color='r')
    #     ax.scatter(X_2[:, 0], X_2[:, 1], color='b')
    #     ax.scatter(X_3[:, 0], X_3[:, 1], color='y')
    #     plt.show()

    x_train = np.concatenate([X_1, X_2, X_3], axis=0)
    y_train = np.concatenate([Y_1, Y_2, Y_3], axis=0)

    # shuffle data
    shuffle_indices = np.random.choice(
        np.arange(len(x_train)),
        size=len(x_train),
        replace=False)
    x_train = torch.tensor(
        x_train[shuffle_indices],
        dtype=torch.float)
    y_train = torch.tensor(
        y_train[shuffle_indices],
        dtype=torch.long)

    target_concentrations = get_target_dirichlet(
        y_train, alpha_0=2)  # new 'labels'

    assert_no_nan_no_inf(x_train)
    assert_no_nan_no_inf(y_train)

    return x_train, target_concentrations, x_train, target_concentrations


def create_loss_fn():
    return kl_loss
    # return torch.nn.CrossEntropyLoss


def create_model():
    model = models.TwoLayer(n_in=2, n_out=3, n_hidden=12)
    return model


def create_optimizer(model):
    for name, param in model.named_parameters():
        print(name, "Requires grad?", param.requires_grad)
    optimizer = torch.optim.Adam(model.parameters())
    return optimizer


def eval_model(model,
               x_test,
               y_test,
               test_loader=None):

    # set model in eval mode
    model.eval()

    # TODO-Add a sequenial test loader
    y_hat, means, alphas, precision = model(x_test)
    assert_no_nan_no_inf(y_hat)
    pred_proba, pred_class = torch.max(y_hat, 1)
    _, true_class = torch.max(y_test, 1)
    accuracy = torch.div(
        (pred_class == true_class).sum().type(torch.float32),
        len(y_test))
    return accuracy, pred_proba, pred_class


def plot_all(model,
             training_loss):

    plot_training_loss(training_loss=training_loss)
    plot_decision_surface(model)


def plot_decision_surface(model):

    # discretize input space i.e. all possible pairs of coordinates
    # between [-40, 40] x [-40, 40]
    possible_vals = np.linspace(-40, 40, 160)
    x_vals, y_vals = np.meshgrid(possible_vals, possible_vals)
    inputs = np.stack((x_vals.flatten(), y_vals.flatten()), axis=1)
    inputs = torch.tensor(inputs, dtype=torch.float32)

    # forward pass model
    y_hat, means, alphas, precision = model(inputs)

    strength_of_mean = torch.max(means, dim=1)[0].detach().numpy()

    plot_data = [
        go.Surface(x=possible_vals,
                   y=possible_vals,
                   z=strength_of_mean.reshape(x_vals.shape))
    ]

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
                y_train):

    # set model in training mode
    model.train()

    # train model
    training_loss = []
    for epoch in range(n_epochs):
        for start_index in range(0, len(x_train), batch_size):
            optimizer.zero_grad()
            x_batch = x_train[start_index:start_index+batch_size]
            y_batch = y_train[start_index:start_index+batch_size]
            y_hat, mean, alphas, precision = model(x_batch)
            assert_no_nan_no_inf(y_hat)
            batch_loss = loss_fn(y_hat, y_batch)
            assert_no_nan_no_inf(batch_loss)
            training_loss.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()
        print("Last obtained batch_loss", batch_loss.item())

    return model, optimizer, training_loss


def get_target_dirichlet(y_train, alpha_0, nb_class=3, epsilon=1e-4):
    """alpha_0  Hyperparameter-> specify the sharpness.
    epsilon ->Smoothing param
    nb_class
    output: target_dirichlet -> torch.distributions.Dirichlet.
    access to concentration parameters with output.concentration"""
    one_hot_labels = F.one_hot(y_train.squeeze()).to(torch.float32)
    soft_labels = one_hot_labels - one_hot_labels * nb_class * epsilon + epsilon
    target_concentrations = alpha_0 * soft_labels
    # target_dirichlet = Dirichlet(target_concentrations)
    return target_concentrations


# TODO: get this working
def kl_loss(model_concentrations, target_concentrations, reverse=True):
    """
    Input: Model softmax outputs or anything else that we want to build
    our Dirichlet distribution on.
    """
    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(
        model_concentrations)  # is that what we want ? Our concentrations parameters. sum to one at the end, no matter what alpha_0
    if reverse:
        kl_divs = _kl_dirichlet_dirichlet(
            p=target_dirichlet, q=model_dirichlet)
    else:  # forward
        kl_divs = _kl_dirichlet_dirichlet(
            p=model_dirichlet, q=target_dirichlet)
    assert_no_nan_no_inf(kl_divs)
    mean_kl = torch.mean(kl_divs)
    return mean_kl
