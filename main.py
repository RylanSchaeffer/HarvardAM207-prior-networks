import utils


def main(args):

    model, optimizer, loss_fn, data = utils.setup(args=args)

    model, optimizer, training_loss = utils.train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        x_train=data['x_train'],
        target_concentrations=data['concentrations_train'])

    accuracy, pred_proba, pred_class = utils.eval_model(
        model=model,
        x_test=data['x_test'],
        y_test=data['concentrations_test'])

    # ood_indices = data['concentrations_test'].sum(1) == 3.

    utils.plot_all(
        x_train=data['x_train'],
        labels_train=data['labels_train'],
        concentrations_train=data['concentrations_train'],
        model=model,
        training_loss=training_loss)


if __name__ == '__main__':
    args = utils.create_args()
    main(args)
