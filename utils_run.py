import plotly.io as pio
from plotly.subplots import make_subplots
import argparse
import logging
import numpy as np
import plotly.graph_objects as go
import torch
from torch.distributions import Categorical, Dirichlet
from torch.distributions.kl import _kl_dirichlet_dirichlet
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import models
from IPython.display import Image

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


def create_optimizer(model, lr=0.001):
    for name, param in model.named_parameters():
        print(name, "Requires grad?", param.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
                     z=1.1 * \
                     np.full(x_train.shape[0], fill_value=np.max(
                         entropy_of_means)),
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
    plot_decision_surface(
        model=model, x_train=train_samples, labels_train=labels_train)


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


def plot_alphas_train(out, nb_classes=3, sharp=101, flat=1, 
renderer='jpeg'):
    #TODO-Remove dirty legend on the side.
    fig = make_subplots(
        rows=nb_classes, cols=nb_classes, shared_xaxes=True, vertical_spacing=0.1, 
        subplot_titles=(f"Target:({sharp}, {flat}, {flat}) 1st coordinate",
                        f"Target:({flat}, {sharp}, {flat})",
                        f"Target:({flat}, {flat}, {sharp})",
                        "2nd coordinate", "", "",
                        "3rd coordinate", "", "")
                        )

    x_axis = np.arange(len(out[0][1][:, 0].detach()))
    #Really dirty double for loops...
    for i in range(nb_classes):
        for j in range(nb_classes):
            if i == j:
                fig.add_trace(
                    go.Scatter(x=x_axis, y=out[i][1][:, i].detach(), 
                                line=dict(color='firebrick')), row=i+1, col=i+1)
                fig.add_trace(
                    go.Scatter(x=x_axis, y=[sharp]*len(x_axis),  
                                line=dict(color='black')), row=i+1, col=i+1)
            else:
                fig.add_trace(
                    go.Scatter(x=x_axis, y=out[i][1][:, j].detach(),
                                line=dict(color='royalblue')), row=i+1, col=j+1)
                fig.add_trace(go.Scatter(
                    x=x_axis, y=[flat]*len(x_axis),  
                        line=dict(color='black')), row=i+1, col=j+1)

        fig.update_yaxes(title_text="alpha value")
        fig.update_xaxes(title_text="gradient step")
        #fig.show(showlegend=False)
        fig.update_layout(height=1000, width=1000,
                          title_text="Output of the network per gradient step ")
    #fig.show(showlegend=False, renderer=renderer)
    fig.show(renderer=renderer, width=1000, height=1000)
    #Image(pio.to_image(fig, format='png'))


def plot_alphas_histogram(out, nb_classes=3,
                          sharp=101, flat=1, renderer='png'):
    #TODO-Remove dirty legend on the side.
    #TODO-Better explain -> legend
    #fig, ax = plt.subplots(3, nb_classes, figsize=(20, 5))
    fig = make_subplots(
        rows=nb_classes, cols=nb_classes, vertical_spacing=0.1, 
        subplot_titles=(f"Target:({sharp}, {flat}, {flat}) 1st coordinate",
                         f"Target:({flat}, {sharp}, {flat})",
                        f"Target:({flat}, {flat}, {sharp})",
                        "2nd coordinate", "", "",
                            "3rd coordinate", "", "")
                            )
    x_axis = np.arange(len(out[0][1][:, 0].detach()))
    #Really dirty double for loops...
    for i in range(nb_classes):
        # Add traces
        for j in range(nb_classes):
            if i == j:
                fig.add_trace(go.Histogram(x=out[i][1][:, i].detach(),
                                           marker=dict(color='FireBrick')), row=i+1, col=i+1)
            else:
                fig.add_trace(go.Histogram(x=out[i][1][:, j].detach(),
                                           marker=dict(color='RoyalBlue')), row=i+1, col=j+1)

        fig.update_yaxes(title_text="count")
        fig.update_xaxes(title_text="alpha value")

        fig.update_layout(height=1000, width=1000,
                          title_text="Distribution of the the network's output")
    #fig.show(showlegend=False)
    #Image(pio.to_image(fig, format='png'))
    #fig.show(renderer=renderer)
    fig.show(renderer=renderer, width=1000, height=1000)


def plot_precision_train(precision_train, sharp=101, flat=1, 
renderer='png'):
    x_axis = np.arange(len(precision_train.reshape(-1)))
    fig=go.Figure(data=go.Scatter(x=x_axis,
    y=precision_train.reshape(-1).detach(), 
    line=dict(color='firebrick')))
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=[sharp+flat+flat] * len(x_axis),
        line=dict(color='black')))

    fig.update_yaxes(title_text="precisoin value")
    fig.update_xaxes(title_text="graident step")

    fig.update_layout(height=1000, width=1000,
                          title_text="Precision of the network per gradient step")

    #fig.show(renderer=renderer)
    fig.show(renderer=renderer, width=1000, height=1000)
    #Image(pio.to_image(fig, format='png'))


def plot_dirichlet_simplex(dirichlet_distrib, subdiv=4, nlevels=500, **kwargs):
    """
    Visualize the distribution of a mixture of Dirichlet on a simplex. 
    INPUT:
        dirichlet distrib: list of torch.distributions.Dirichlet objects (or other distributions)
        subdiv : plotting parameters(finer discretization)
        nlevels : plotting parameters (higher means more spread)
    OUT:
        None
        return a plot of the SUM/MEAN of the specified dirichlet_distrib's pdf.
    """
    assert isinstance(dirichlet_distrib, list), "Need to provide a list of torch.distributions objects - if only one\
        provide a list of length 1"

    def pdf(dirichlet_dist, x):
        #Pdf of Dirichlet or any other distrib actually. Out: torch.Tensor
        def pdf(x): return torch.exp(dirichlet_dist.log_prob(x))
        out_pdf = pdf(x) if isinstance(
            x, torch.Tensor) else pdf(torch.tensor(x))
        return out_pdf

    def xy2bc(xy, corners, midpoints, tol=1.e-3):
        #Converts 2D-Cartesian coordinates to barycentric. Out:array
        s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75
             for i in range(3)]
        return np.clip(s, tol, 1.0 - tol)

    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    # Mid-points of triangle sides opposite of each corner
    midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0
                 for i in range(3)]

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    #pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    bar_points = [xy2bc(xy, corners, midpoints)
                  for xy in zip(trimesh.x, trimesh.y)]
    #pvals = [pdf(dirichlet_distrib, xy2bc(xy, corners, midpoints)) for xy in zip(trimesh.x, trimesh.y)]
    pvals = np.zeros((len(dirichlet_distrib), len(bar_points))
                     )  # nb of distributions, nb of points
    for i, dist in enumerate(dirichlet_distrib):
        pvals[i] = pdf(dist, bar_points).numpy()
    pvals = pvals.sum(axis=0)  # Remove the first axis
    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')


def plot_dirichlet_model(model, data, *args, **kwargs):
    """
    Visualize the dirichlet distribution induced by the data.
    Mixture weights are from the empirical distribution of the data.
    See paper.
    INPUT: Trained model, data
    """
    model.eval()
    _, _, alphas, precision = model(data['samples'])
    alphas = alphas.detach()  # To visualize with numpy-detach from graph
    dirichlet_distrib = [torch.distributions.Dirichlet(alphas[i]) for i in range(
        len(alphas))]  # Define Dirichlet form the parameters
    plot_dirichlet_simplex(dirichlet_distrib, *args, **kwargs)
    

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
                target_concentrations,
                verbose=True,
                track_mode=False):
    # TODO: Keep track of the model_concentrations parameters. Maybe an output dictionnary so that we can
    # keep track or more/less things that we want to investigate, e.g:
    # tracks = {'training_loss':[], 'model_concentrations':[], ...}
    #trackers = {track_name: [] for track_name in track}
    # set model in training mode
    """
    track_mode: bool
        By specifying track_mode, we also output a dictionnary 
        keeping track of the concentrations/precision. For post training analaysis. 
    """
    model.train()
    tracks = {'model_concentrations': [],
              'target_concentrations': [], 'precision': []}

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
            logits, mean, model_concentrations, precision = model(
                x_train_batch)
            assert_no_nan_no_inf(model_concentrations)
            batch_loss = loss_fn(model_concentrations,
                                 target_concentrations_batch)
            assert_no_nan_no_inf(batch_loss)
            training_loss.append(batch_loss.item())
            if track_mode:
                tracks['model_concentrations'].append(model_concentrations)
                tracks['target_concentrations'].append(
                    target_concentrations_batch)
                tracks['precision'].append(precision)
            batch_loss.backward()
            optimizer.step()

        if verbose:
            print("Last obtained batch_loss", batch_loss.item())

    if track_mode:
        tracks['training_loss'] = training_loss
        return model, optimizer, tracks
    return model, optimizer, training_loss
