import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import torch

import utils.measures


def plot_decision_surface(model,
                          samples,
                          labels,
                          labels_names,
                          z_fn,
                          x_axis_title,
                          y_axis_title,
                          z_axis_title,
                          possible_xvals=np.linspace(-40, 40, 81),
                          possible_yvals=np.linspace(-35, 25, 81)):

    # discretize input space i.e. all possible pairs of coordinates
    x_vals, y_vals = np.meshgrid(possible_xvals, possible_yvals)
    grid_inputs = np.stack((x_vals.flatten(), y_vals.flatten()), axis=1)
    grid_inputs = torch.tensor(grid_inputs, dtype=torch.float32)

    # forward pass model
    model_outputs = model(grid_inputs)

    z_vals = z_fn(categorical_parameters=model_outputs['mean'])
    max_z_val = np.max(z_vals)

    # create decision surface
    traces = [go.Surface(
        x=possible_xvals,
        y=possible_yvals,
        z=z_vals.reshape(x_vals.shape),
        showscale=False)]

    # add plot
    unique_labels = np.unique(labels)
    for unique_label, label_name in zip(unique_labels, labels_names):
        # add training points
        scatter_trace = go.Scatter3d(x=samples[labels == unique_label, 0],
                     y=samples[labels == unique_label, 1],
                     name=label_name,
                     z=1.1 * np.full(samples.shape[0], fill_value=max_z_val),
                     mode='markers',
                     marker=dict(color=unique_label))
        traces.append(scatter_trace)

    layout = dict(
        title='Decision Surface',
        scene=dict(
            zaxis=dict(title=z_axis_title),
            xaxis=dict(title=x_axis_title),
            yaxis=dict(title=y_axis_title)
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


# def plot_decision_surface_LR(model,
#                              x_train,
#                              labels_train):
#     # discretize input space i.e. all possible pairs of coordinates
#     # between [-40, 40] x [-40, 40]
#     possible_vals = np.linspace(-40, 40, 81)
#     x_vals, y_vals = np.meshgrid(possible_vals, possible_vals)
#     grid_inputs = np.stack((x_vals.flatten(), y_vals.flatten()), axis=1)
#     grid_inputs = torch.tensor(grid_inputs, dtype=torch.float32)
#
#     # forward pass model
#     y_hat = model(grid_inputs)
#
#     entropy = utils.measures.entropy_categorical(
#         categorical_parameters=y_hat)
#
#     plot_data = [
#         # add model outputs
#         go.Surface(x=possible_vals,
#                    y=possible_vals,
#                    z=entropy.reshape(x_vals.shape)),
#         # add training points
#         go.Scatter3d(x=x_train[:, 0],
#                      y=x_train[:, 1],
#                      z=1.1 * np.full(x_train.shape[0], fill_value=np.max(entropy)),
#                      mode='markers',
#                      marker=dict(color=labels_train))
#     ]
#
#     layout = dict(
#         title='Decision Surface',
#         scene=dict(
#             zaxis=dict(title='Entropy of predictions'),
#             xaxis=dict(title='input_dim_1'),
#             yaxis=dict(title='input_dim_2')
#         )
#     )
#
#     fig = go.Figure(data=plot_data, layout=layout)
#     fig.show()


def plot_MI(model,
            samples,
            labels,
            labels_names,
            x_axis_title,
            y_axis_title,
            possible_xvals=np.linspace(-40, 40, 81),
            possible_yvals=np.linspace(-35, 25, 81)):

    # discretize input space i.e. all possible pairs of coordinates
    x_vals, y_vals = np.meshgrid(possible_xvals, possible_yvals)
    grid_inputs = np.stack((x_vals.flatten(), y_vals.flatten()), axis=1)
    grid_inputs = torch.tensor(grid_inputs, dtype=torch.float32)

    # forward pass model
    model_outputs = model(grid_inputs)
    alphas = model_outputs['concentrations']

    mi = []
    for i in range(len(alphas.detach().numpy())):
        mi.append(utils.measures.mutual_info_dirichlet(dirichlet_concentrations=alphas[i]))
    mi = np.array(mi)

    z_vals = mi
    max_z_val = np.max(z_vals)

    # create decision surface
    traces = [go.Surface(
        x=possible_xvals,
        y=possible_yvals,
        z=z_vals.reshape(x_vals.shape),
        showscale=False)]

    # add plot
    unique_labels = np.unique(labels)
    for unique_label, label_name in zip(unique_labels, labels_names):
        # add training points
        scatter_trace = go.Scatter3d(x=samples[labels == unique_label, 0],
                     y=samples[labels == unique_label, 1],
                     name=label_name,
                     z=1.1 * np.full(samples.shape[0], fill_value=max_z_val),
                     mode='markers',
                     marker=dict(color=unique_label))
        traces.append(scatter_trace)

    layout = dict(
        title='Decision Surface',
        scene=dict(
            zaxis=dict(title="Mutual Information"),
            xaxis=dict(title=x_axis_title),
            yaxis=dict(title=y_axis_title)
        )
    )


    fig = go.Figure(data=traces, layout=layout)
    fig.show()

"""
def plot_bound_MI(model,
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
    mi = []
    for i in range(len(alphas.detach().numpy())):
        mi.append(mutual_information(dirichlet_concentrations=alphas[i]))
    mi = np.array(mi)

    plot_data = [
        # add model outputs
        go.Surface(x=possible_vals,
                   y=possible_vals,
                   z=mi.reshape(x_vals.shape)),
        # add training points
        go.Scatter3d(x=x_train[:, 0],
                     y=x_train[:, 1],
                     z=1.1 * np.full(x_train.shape[0], fill_value=np.max(mi)),
                     mode='markers',
                     marker=dict(color=labels_train))
    ]

    layout = dict(
        title='Decision Surface',
        scene=dict(
            zaxis=dict(title='Mutual Information'),
            xaxis=dict(title='input_dim_1'),
            yaxis=dict(title='input_dim_2')
        )
    )

    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()
"""

def plot_results(train_samples,
                 labels_train,
                 train_concentrations,
                 model,
                 training_loss):

    plot_training_data(samples=train_samples, labels=labels_train)
    plot_training_loss(training_loss=training_loss)
    plot_decision_surface(model=model, samples=train_samples, labels=labels_train)
    plot_MI(model=model, x_train=train_samples, labels_train=labels_train)


def plot_training_data(samples,
                       labels,
                       labels_names=None,
                       plot_title='Training Data',
                       xaxis=None,
                       yaxis=None):
    if xaxis is None:
        xaxis = dict(title='x')
    if yaxis is None:
        yaxis = dict(title='y')

    unique_labels = np.unique(labels)

    traces = []
    if labels_names is None:
        labels_names = [f'Class {i}' for i in range(len(unique_labels))]
    #else:
        #assert len(labels_names)==len(unique_labels)
    
    for unique_label, label_name in zip(unique_labels, labels_names):
        trace = go.Scatter(
            x=samples[labels == unique_label, 0],
            y=samples[labels == unique_label, 1],
            name=label_name,
            mode='markers',
            marker=dict(color=unique_label),
        )
        traces.append(trace)

    layout = dict(
        title=plot_title,
        xaxis=xaxis,
        yaxis=yaxis)

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_training_loss(training_loss,
                       plot_title='Training Loss per Gradient Step',
                       xaxis=dict(title='Gradient Step'),
                       yaxis=dict(title='Training Loss')):
    trace = go.Scatter(
        x=np.arange(len(training_loss)),
        y=training_loss)
    layout = dict(
        title=plot_title,
        xaxis=xaxis,
        yaxis=yaxis)
    fig = go.Figure(data=trace, layout=layout)
    fig.show()

#TODO-Turn into a utils/data function.
def get_sharp_indices(target_conc_train, model_conc_train, sharp=101, flat=1):
    """
    Inputs:
        -target_conc_train: tensors: (total nb of batches in training, batch_size, nb_classes)
            The tensor of the target concentrations used for training the model. Obtained form train model.
        -model_conc_train: Similar.
    Outputs:
        -outputs: list, length=nb_classes.
                l[0] = tuple of size 2
            For each class, gives the corresponding points in the training data
            AND the output of the network for these same points.
    """
    nb_classes = target_conc_train.shape[-1]
    outputs = []
    for i in range(nb_classes):
        # The data points for which, class i.
        sharp_indices = (target_conc_train[:, :, i] == sharp)
        outputs.append((sharp_indices, model_conc_train[sharp_indices]))
    return outputs


def plot_alphas_train(out, nb_classes=3, sharp=101, flat=1,
                      renderer='jpeg'):
    #TODO-Remove dirty legend on the side.
    #out refers to the output of  get_sharp_indices
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
    fig.update_layout(showlegend=False)
    fig.show(renderer=renderer, width=1000, height=1000)
    #Image(pio.to_image(fig, format='png'))


def plot_alphas_histogram(out, nb_classes=3,
                          sharp=101, flat=1, renderer='png'):
    #TODO-Better explain -> legend ?
    #out refers to the output of  get_sharp_indices
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
    fig.update_layout(showlegend=False) #Remove the dirty unnamed traces
    fig.show(renderer=renderer, width=1000, height=1000)


def plot_precision_train(precision_train, sharp=101, flat=1,
                         renderer='png'):
    x_axis = np.arange(len(precision_train.reshape(-1)))
    fig = go.Figure(data=go.Scatter(x=x_axis,
                                    y=precision_train.reshape(-1).detach(),
                                    line=dict(color='firebrick')))

    fig.add_trace(go.Scatter(
        x=x_axis, y=[sharp+flat+flat] * len(x_axis),
        line=dict(color='black')))

    fig.update_yaxes(title_text="precision value")
    fig.update_xaxes(title_text="gradient step")

    fig.update_layout(height=1000, width=1000,
                      title_text="Precision of the network per gradient step")

    #fig.show(renderer=renderer)
    fig.update_layout(showlegend=False)
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
    model_outputs = model(data['samples'])
    alphas = model_outputs['concentrations']
    alphas = alphas.detach()  # To visualize with numpy-detach from graph
    dirichlet_distrib = [torch.distributions.Dirichlet(alphas[i]) for i in range(
        len(alphas))]  # Define Dirichlet form the parameters
    plot_dirichlet_simplex(dirichlet_distrib, *args, **kwargs)
