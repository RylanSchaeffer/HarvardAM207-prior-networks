import numpy as np
import plotly.graph_objects as go
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
                       labels_names,
                       plot_title='Training Data',
                       xaxis=None,
                       yaxis=None):
    if xaxis is None:
        xaxis = dict(title='x')
    if yaxis is None:
        yaxis = dict(title='y')

    unique_labels = np.unique(labels)

    traces = []
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
