import numpy as np
import torch


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def create_data():
    """
    - Args to specify the mean variances if need be, under the form mu_1, sigma_1" and so on
    output: np.array with samples from each gaussian.
    """
    means = np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.],
        [-4., -4.]])
    means *= 10

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
    assert torch.all(concentrations_train > 0)

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
