from utils import data, plot, run, measures
import numpy as np


def main(args):

    train_data = data.create_data(
        create_data_functions=[
            data.create_data_mixture_of_gaussians,
        ],
        functions_args=[
            data.mog_three_in_distribution_one_out
        ])

    model, optimizer, loss_fn = run.setup(
        args=args,
        in_dim=train_data['samples'].shape[1],
        out_dim=train_data['concentrations'].shape[1])

    model, optimizer, training_loss = run.train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=args['n_epochs'],
        batch_size=args['batch_size'],
        train_data=train_data,
        args=args)

    # accuracy, pred_proba, pred_class = run.eval_model(
    #     model=model,
    #     x_test=test_data['samples'],
    #     y_test=test_data['concentrations'])
    #
    plot.plot_results(
        train_samples=train_data['samples'],
        labels_train=train_data['targets'],
        train_concentrations=train_data['concentrations'],
        model=model,
        training_loss=training_loss)


if __name__ == '__main__':
    args = run.create_args()
    # convert args from a namespace to a dict
    args = vars(args)
    main(args)
