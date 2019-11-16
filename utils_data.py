import numpy as np
import torch


mog_three_in_distribution = {
    'means': 5 * np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.]]),
    'covariances': np.array([
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
    ]),
    'n_samples_per_gaussian': np.array(
        [100, 100, 100]),
    'out_of_distribution': np.array(
        [False, False, False])
}


mog_ood_in_middle_no_overlap = {
    'means': 5 * np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.],
        [0., 0.]]),
    'covariances': np.array([
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
    ]),
    'n_samples_per_gaussian': np.array(
        [100, 100, 100, 100]),
    'out_of_distribution': np.array(
        [False, False, False, True])
}


mog_ood_in_middle_overlap = {
    'means': 2 * np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.],
        [0., 0.]]),
    'covariances': np.array([
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
    ]),
    'n_samples_per_gaussian': np.array(
        [100, 100, 100, 100]),
    'out_of_distribution': np.array(
        [False, False, False, True])
}


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def create_data(create_data_functions,
                functions_args):
    """
    create_data_functions is a list of create_data_placeholder functions,
    specifying the
    output: np.array with samples from each gaussian.
    """

    for i, (create_data_function, function_kwargs) in \
            enumerate(zip(create_data_functions, functions_args)):

        if i == 0:
            samples, targets, concentrations = create_data_function(
                **function_kwargs)
        else:
            new_samples, new_targets, new_concentrations = create_data_function(
                **function_kwargs)

            # append new samples to previous data
            samples = np.concatenate((samples, new_samples), axis=0)
            targets = np.concatenate((targets, new_targets), axis=0)
            concentrations = np.concatenate((concentrations, new_concentrations), axis=0)

    # shuffle data
    n_total_samples = samples.shape[0]
    shuffle_indices = np.random.choice(
        np.arange(n_total_samples),
        size=n_total_samples,
        replace=False)
    samples = samples[shuffle_indices]
    targets = targets[shuffle_indices]
    concentrations = concentrations[shuffle_indices]

    # convert arrays to tensors
    samples = torch.tensor(
        samples,
        dtype=torch.float)
    targets = torch.tensor(
        targets,
        dtype=torch.long)
    concentrations = torch.tensor(
        concentrations,
        dtype=torch.float32)

    # check values are valid
    assert_no_nan_no_inf(samples)
    assert_no_nan_no_inf(targets)
    assert_no_nan_no_inf(concentrations)
    assert torch.all(concentrations > 0)

    # group and return data
    data = dict(
        samples=samples,
        targets=targets,
        concentrations=concentrations)

    return data


def create_data_mixture_of_gaussians(means,
                                     covariances,
                                     n_samples_per_gaussian,
                                     out_of_distribution):
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
    :param out_of_distribution: bool,  shape (number of gaussians, )
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
    n_gaussians = (~out_of_distribution).sum()
    mog_concentrations = np.ones((n_total_samples, n_gaussians))
    in_distribution_rows = np.isin(
        mog_labels,
        np.argwhere(~out_of_distribution))
    mog_concentrations[in_distribution_rows, mog_labels[in_distribution_rows]] += 100

    return mog_samples, mog_labels, mog_concentrations


def create_data_ring(prev_samples=None,
                     prev_labels=None,
                     prev_concentrations=None):
    # append new samples to previous data, if provided
    if prev_samples is not None:
        # check that other arrays are also not None
        assert prev_labels is not None
        assert prev_concentrations is not None

        mog_samples = np.concatenate((mog_samples, prev_samples), axis=0)
        mog_labels = np.concatenate((mog_labels, prev_labels), axis=0)
        mog_concentrations = np.concatenate((mog_concentrations, prev_concentrations), axis=0)

    return mog_samples, mog_labels, mog_concentrations
