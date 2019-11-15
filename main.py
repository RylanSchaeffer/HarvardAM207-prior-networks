import utils_data
import utils_run


def main(args):

    data = utils_data.create_data()

    model, optimizer, loss_fn = utils_run.setup(
        args=args,
        data=data)

    model, optimizer, training_loss = utils_run.train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        x_train=data['x_train'],
        target_concentrations=data['concentrations_train'])

    accuracy, pred_proba, pred_class = utils_run.eval_model(
        model=model,
        x_test=data['x_test'],
        y_test=data['concentrations_test'])

    # ood_indices = data['concentrations_test'].sum(1) == 3.

    utils_run.plot_all(
        x_train=data['x_train'],
        labels_train=data['labels_train'],
        concentrations_train=data['concentrations_train'],
        model=model,
        training_loss=training_loss)


if __name__ == '__main__':
    args = utils_run.create_args()
    main(args)
