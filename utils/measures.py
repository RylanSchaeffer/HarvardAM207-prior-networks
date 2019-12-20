import numpy as np
import torch
from torch.distributions import Categorical, Dirichlet
from torch.distributions.kl import _kl_dirichlet_dirichlet
from torch.nn import NLLLoss
from scipy import special


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def kl_divergence(model_concentrations,
                  target_concentrations,
                  mode='reverse'):
    """
    Input: Model concentrations, target concentrations parameters.
    Output: Average of the KL between the two Dirichlet.
    """
    assert torch.all(model_concentrations > 0)
    assert torch.all(target_concentrations > 0)

    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(model_concentrations)
    kl_divergences = _kl_dirichlet_dirichlet(
        p=target_dirichlet if mode == 'forward' else model_dirichlet,
        q=model_dirichlet if mode == 'forward' else target_dirichlet)
    assert_no_nan_no_inf(kl_divergences)
    mean_kl = torch.mean(kl_divergences)
    assert_no_nan_no_inf(mean_kl)
    return mean_kl


def kl_loss_fn(loss_input,
               mode='reverse'):

    model_concentrations = loss_input['model_outputs']['concentrations']
    target_concentrations = loss_input['y_concentrations_batch']
    loss = kl_divergence(
        model_concentrations=model_concentrations,
        target_concentrations=target_concentrations,
        mode=mode)
    assert_no_nan_no_inf(loss)
    return loss


def neg_log_likelihood(input,
                       target):

    nll_fn = NLLLoss()
    nll = nll_fn(input=input, target=target)
    assert_no_nan_no_inf(nll)
    return nll


def nll_loss_fn(loss_inputs):
    y_pred_batch = loss_inputs['model_outputs']['y_pred']
    y_batch = loss_inputs['y_batch']
    loss = neg_log_likelihood(
        input=y_pred_batch,
        target=y_batch)
    assert_no_nan_no_inf(loss)
    return loss


def entropy_categorical(categorical_parameters):
    entropy = Categorical(categorical_parameters).entropy()
    # TODO: discuss whether we want numpy in these functions
    assert_no_nan_no_inf(entropy)
    entropy = entropy.detach().numpy()
    return entropy


def entropy_dirichlet(dirichlet_concentrations):
    entropy = Dirichlet(dirichlet_concentrations).entropy()
    # TODO: discuss whether we want numpy in these functions
    entropy = entropy.detach().numpy()
    assert_no_nan_no_inf(entropy)
    return entropy


def mutual_info_dirichlet(dirichlet_concentrations):
    # TODO: discuss whether we want numpy in these functions
    dirichlet_concentrations = dirichlet_concentrations.detach().numpy()
    dirichlet_concentrations_sum = dirichlet_concentrations.sum()
    res = (1.0/dirichlet_concentrations_sum)*dirichlet_concentrations*(np.log(dirichlet_concentrations*1.0/dirichlet_concentrations_sum)-special.digamma(dirichlet_concentrations+1)+special.digamma(dirichlet_concentrations_sum+1))
    final_res = res.sum() * (-1.0)
    #assert_no_nan_no_inf(final_res)
    return final_res
