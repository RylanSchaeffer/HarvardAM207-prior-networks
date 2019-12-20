import numpy as np
from scipy.linalg import block_diag
import torch


mog_three_in_distribution = {
    'gaussians_means': 5 * np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.]]),
    'gaussians_covariances': np.array([
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
    ]),
    'n_samples_per_gaussian': np.array(
        [100, 100, 100]),
    'out_of_distribution': np.array(
        [False, False, False])
}

mog_three_in_distribution_one_out = {
    'gaussians_means': 5 * np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.],
        [-5., -5.]]
    ),
    'gaussians_covariances': np.array([
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


mog_three_in_distribution_overlap = {
    'gaussians_means': 5 * np.array([
        [0., 1.],
        [-1, 0.],
        [1, 0]]),
    'gaussians_covariances': np.array([
        [[4, 0], [0, 4]],
        [[4, 0], [0, 4]],
        [[4, 0], [0, 4]],
    ]),
    'n_samples_per_gaussian': np.array(
        [100, 100, 100]),
    'out_of_distribution': np.array(
        [False, False, False])
}


mog_ood_in_middle_no_overlap = {
    'gaussians_means': 5 * np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.],
        [0., 0.]]),
    'gaussians_covariances': np.array([
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
    ]),
    'n_samples_per_gaussian': np.array(
        [100, 100, 100, 500]),
    'out_of_distribution': np.array(
        [False, False, False, True])
}


mog_ood_in_middle_overlap = {
    'gaussians_means': 2 * np.array([
        [0., 2.],
        [-np.sqrt(3), -1.],
        [np.sqrt(3), -1.],
        [0., 0.]]),
    'gaussians_covariances': np.array([
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
        [[2.0, 0], [0, 2.0]],
    ]),
    'n_samples_per_gaussian': np.array([
        100, 100, 100, 100]),
    'out_of_distribution': np.array([
        False, False, False, True])
}

rings = {
    'centers': np.array([
        #[0., 0.],
        [0., 0.]
    ]),
    'inner_radii': np.array([
        #[0.],
        [20.]
    ]),
    'outer_radii': np.array([
        #[20],
        [25.]
    ]),
    'n_samples_per_shell': np.array([
        #100,
        1000
    ]),
    'out_of_distribution': np.array([
        #False,
        True
    ])
}


parallelepipeds = {
    'parallelepiped_centers': np.array([
        [2., 2.],
        [-4., -4.]
    ]),
    'skew_matrices': np.array([
        [[2.0, 2.0], [0, 1.0]],
        [[-1.0, 0], [-2.0, -2.0]],
    ]),
    'n_samples_per_parallelepiped': np.array([
        100,
        100
    ]),
    'out_of_distribution': np.array([
        False,
        False
    ])
}

parallelepipeds_ood_in_between = {
    'parallelepiped_centers': np.array([
        [2., 2.],
        [-4., -4.],
        [-1., -1.]
    ]),
    'skew_matrices': np.array([
        [[2.0, 2.0], [0, 1.0]],
        [[-1.0, 0], [-2.0, -2.0]],
        [[2.0, 2.0], [-2.0, -2.0]]
    ]),
    'n_samples_per_parallelepiped': np.array([
        100,
        100,
        100
    ]),
    'out_of_distribution': np.array([
        False,
        False,
        True
    ])
}


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def create_data(create_data_functions,
                functions_args):
    """
    We want the ability to chain together sequences of create_data_function
    functions. Consequently, we need to do some bookkeeping to that
    we correctly combine labels and concentrations.

    For labels, each create_data_function creates classes numbered from
    0, 1, ..., n-1. Thus, after each function call, we add 1 + largest
    class label to the new labels to ensure each new label is unique.



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

            # append new samples to previous samples
            samples = np.concatenate((samples, new_samples), axis=0)

            # append new targets to previous targets
            # shift new targets to not overlap with previous targets
            new_targets += np.max(targets) + 1
            targets = np.concatenate((targets, new_targets), axis=0)

            # append new concentrations to previous concentrations
            # can't forget to set off-diagonal 0s to 1!
            concentrations = block_diag(concentrations, new_concentrations)
            concentrations[concentrations == 0.] = 1.

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


def create_data_parallelepipeds(parallelepiped_centers,
                                skew_matrices,
                                n_samples_per_parallelepiped,
                                out_of_distribution):

    dim_of_parallelepiped = parallelepiped_centers.shape[1]
    n_total_samples = n_samples_per_parallelepiped.sum()

    # preallocate arrays to store new samples
    parallelepipeds_samples = np.zeros(shape=(n_total_samples, dim_of_parallelepiped))
    parallelepipeds_targets = np.zeros(shape=(n_total_samples,), dtype=np.int)

    write_index = 0
    for i, (center, skew_matrix, n_samples) in \
            enumerate(zip(parallelepiped_centers, skew_matrices, n_samples_per_parallelepiped)):

        # generate and save samples
        uniform_samples = np.random.uniform(
            low=-1, high=1, size=(n_samples, dim_of_parallelepiped))
        parallelepiped_samples = center + np.matmul(skew_matrix, uniform_samples.T).T
        parallelepipeds_samples[write_index:write_index + n_samples] = \
            parallelepiped_samples

        # generate and save labels
        parallelepiped_targets = np.full(shape=n_samples, fill_value=i)
        parallelepipeds_targets[write_index:write_index + n_samples] = \
            parallelepiped_targets

        write_index += n_samples

    # generate concentrations
    n_parallelepipeds = (~out_of_distribution).sum()
    parallelepipeds_concentrations = np.ones((n_total_samples, n_parallelepipeds))
    in_distribution_rows = np.isin(
        parallelepipeds_targets,
        np.argwhere(~out_of_distribution))
    parallelepipeds_concentrations[in_distribution_rows, parallelepipeds_targets[in_distribution_rows]] += 100

    return parallelepipeds_samples, parallelepipeds_targets, parallelepipeds_concentrations


def create_data_mixture_of_gaussians(gaussians_means,
                                     gaussians_covariances,
                                     n_samples_per_gaussian,
                                     out_of_distribution):
    """
    Creates a mixture of Gaussians with corresponding labels and Dirichlet
    concentration parameters. If x_data and y_data are given, append the new
    samples to x_data, y_data and return those instead.

    :param prev_samples:
    :param prev_concentrations:
    :param prev_labels:
    :param gaussians_means: shape (number of gaussians, dim of gaussian)
    :param gaussians_covariances: shape (number of gaussians, dim of gaussian, dim_of_gaussian)
    :param n_samples_per_gaussian: int, shape (number of gaussians, )
    :param out_of_distribution: bool,  shape (number of gaussians, )
    :return mog_samples: shape (sum(n_samples_per_gaussian), dim of gaussian)
    :return mog_targets: int, shape (sum(n_samples_per_gaussian), )
    :return mog_concentrations: shape (sum(n_samples_per_gaussian), dim of gaussian)
    """

    dim_of_gaussians = gaussians_means.shape[1]
    n_total_samples = n_samples_per_gaussian.sum()

    # preallocate arrays to store new samples
    mog_samples = np.zeros(shape=(n_total_samples, dim_of_gaussians))
    mog_targets = np.zeros(shape=(n_total_samples,), dtype=np.int)

    write_index = 0
    for i, (mean, covariance, n_samples) in \
            enumerate(zip(gaussians_means, gaussians_covariances, n_samples_per_gaussian)):
        # generate and save samples
        gaussian_samples = np.random.multivariate_normal(
            mean=mean, cov=covariance, size=n_samples)
        mog_samples[write_index:write_index + n_samples] = gaussian_samples

        # generate and save labels
        gaussian_targets = np.full(shape=n_samples, fill_value=i)
        mog_targets[write_index:write_index + n_samples] = gaussian_targets

        write_index += n_samples

    # generate concentrations
    n_gaussians = (~out_of_distribution).sum()
    mog_concentrations = np.ones((n_total_samples, n_gaussians))
    in_distribution_rows = np.isin(
        mog_targets,
        np.argwhere(~out_of_distribution))
    mog_concentrations[in_distribution_rows, mog_targets[in_distribution_rows]] += 100

    return mog_samples, mog_targets, mog_concentrations


def create_data_spherical_shells(centers,
                                 inner_radii,
                                 outer_radii,
                                 n_samples_per_shell,
                                 out_of_distribution):

    """

    :param n_samples_per_shell:
    :param outer_radii:
    :param centers:
    :param inner_radii:
    :param center:
    :param inner_radius:
    :param outer_radius:
    :param n_samples:
    :param out_of_distribution:
    :return:
    """

    # dimension of rings
    dim_spherical_shells = centers.shape[1]
    n_total_samples = n_samples_per_shell.sum()

    # preallocate arrays to store new samples
    shells_samples = np.zeros(shape=(n_total_samples, dim_spherical_shells))
    shells_targets = np.zeros(shape=(n_total_samples,), dtype=np.int)

    write_index = 0
    for i, (center, inner_radius, outer_radius, n_samples) in \
            enumerate(zip(centers, inner_radii, outer_radii, n_samples_per_shell)):

        # generate and save samples
        # we use the property that samples from a multivariate normal
        # are uniformly on the shell when divided by their absolute values
        gaussian_samples = np.random.multivariate_normal(
            mean=np.zeros(dim_spherical_shells),
            cov=np.identity(dim_spherical_shells),
            size=n_samples)
        shell_samples = np.divide(
            gaussian_samples,
            np.linalg.norm(gaussian_samples, axis=1, keepdims=True))

        # sample ra
        shell_radii = np.random.uniform(
            low=inner_radius,
            high=outer_radius,
            size=(n_samples, 1))  # need 1 for broadcasting to agree
        shell_samples = np.multiply(shell_samples, shell_radii)
        shell_samples += center

        shells_samples[write_index:write_index + n_samples] = shell_samples

        # generate and save labels
        shells_targets[write_index:write_index + n_samples] = i

        write_index += n_samples

    # generate concentrations
    n_shells = (~out_of_distribution).sum()
    shells_concentrations = np.ones((n_total_samples, n_shells))
    in_distribution_rows = np.isin(
        shells_targets,
        np.argwhere(~out_of_distribution))
    shells_concentrations[in_distribution_rows, shells_targets[in_distribution_rows]] += 100

    return shells_samples, shells_targets, shells_concentrations
