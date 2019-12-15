import argparse
import numpy as np
import torch

from utils import measures, models


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def create_args():
    parser = create_arg_parser()
    args = parser.parse_args(args=[])  # need this to work in jupyter notebooks
    return args


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_fn_str',
                        help='Loss function to use. Should be either \'nll\' or \'kl\'.',
                        type=str,
                        default='kl')
    parser.add_argument('--reverse',
                        help='Only used if loss is KL divergence. If True, use'
                             ' Backward KL. If unspecified, use Forward KL',
                        action='store_true',
                        default=True)
    parser.add_argument('--n_epochs',
                        help='Number of epochs to train',
                        type=int,
                        default=1000)
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


def create_loss_fn(loss_fn_str,
                   args):

    if loss_fn_str == 'nll':
        loss_fn = measures.nll_loss_fn
    elif loss_fn_str == 'kl':
        loss_fn = measures.kl_loss_fn
    else:
        raise NotImplementedError('Loss function {} not implemented!'.format(loss_fn_str))
    return loss_fn


def create_model(in_dim,
                 out_dim,
                 args,
                 n_per_hidden_layer=()):

    model = models.FeedforwardNetwork(
        n_in=in_dim,
        n_out=out_dim,
        n_per_hidden_layer=n_per_hidden_layer)
    return model


def create_optimizer(model,
                     args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
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


def setup(in_dim,
          out_dim,
          args):

    model = create_model(
        in_dim=in_dim,
        out_dim=out_dim,
        args=args)
    optimizer = create_optimizer(
        model=model,
        args=args)
    loss_fn = create_loss_fn(
        loss_fn_str=args['loss_fn_str'],
        args=args)
    return model, optimizer, loss_fn


def train_model(model,
                optimizer,
                loss_fn,
                n_epochs,
                batch_size,
                train_data,
                args):

    # TODO: Keep track of the model_concentrations parameters. Maybe an output dictionnary so that we can
    # keep track or more/less things that we want to investigate, e.g:
    # tracks = {'training_loss':[], 'model_concentrations':[], ...}

    # set model in training mode
    model.train()

    torch.manual_seed(0)

    # extract data
    x = train_data['samples']
    y = train_data['targets']
    y_concentrations = train_data['concentrations']

    # train model
    training_loss = []
    num_samples = x.shape[0]
    for epoch in range(n_epochs):
        for _ in range(num_samples // batch_size):

            # randomly sample indices for batch
            batch_indices = np.random.choice(
                np.arange(x.shape[0]),
                size=batch_size,
                replace=False)
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]
            y_concentrations_batch = y_concentrations[batch_indices]

            optimizer.zero_grad()
            model_outputs = model(x_batch)

            loss_inputs = {
                'model_outputs': model_outputs,
                'x_batch': x_batch,
                'y_batch': y_batch,
                'y_concentrations_batch': y_concentrations_batch,
            }

            batch_loss = loss_fn(loss_inputs)
            assert_no_nan_no_inf(batch_loss)
            training_loss.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()

    return model, optimizer, training_loss
