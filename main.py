import utils_data
import utils_run


def main(args):

    train_data = utils_data.create_data(
        create_data_functions=[
            utils_data.create_data_parallelepipeds,
            #utils_data.create_data_spherical_shells,
            #utils_data.create_data_mixture_of_gaussians,
            # utils_data.create_data_mixture_of_gaussians,
        ],
        functions_args=[
            # utils_data.parallelepipeds,
            #vutils_data.rings,
            #utils_data.mog_ood_in_middle_overlap,
            # utils_data.mog_three_in_distribution,
            # utils_data.mog_ood_in_middle_no_overlap
            utils_data.parallelepipeds_ood_in_between
            #utils_data.mog_three_in_distribution_overlap
        ])

    # test_data = utils_data.create_data(
    #     create_data_functions=[
    #         utils_data.create_data_spherical_shells,
    #         # utils_data.create_data_mixture_of_gaussians,
    #         utils_data.create_data_mixture_of_gaussians,
    #     ],
    #     functions_args=[
    #         utils_data.rings,
    #         # utils_data.mog_ood_in_middle_overlap,
    #         utils_data.mog_three_in_distribution,
    #     ])
    test_data = train_data

    model, optimizer, loss_fn = utils_run.setup(
        args=args,
        in_dim=train_data['samples'].shape[1],
        out_dim=train_data['concentrations'].shape[1])

    model, optimizer, training_loss = utils_run.train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        x_train=train_data['samples'],
        target_concentrations=train_data['concentrations'])

    accuracy, pred_proba, pred_class = utils_run.eval_model(
        model=model,
        x_test=test_data['samples'],
        y_test=test_data['concentrations'])

    utils_run.plot_results(
        train_samples=train_data['samples'],
        labels_train=train_data['targets'],
        train_concentrations=train_data['concentrations'],
        model=model,
        training_loss=training_loss)


if __name__ == '__main__':
    args = utils_run.create_args()
    main(args)
