import utils


def main(args):

    model, optimizer, loss_fn, data = utils.setup(args=args)

    x_train, y_train, x_test, y_test = data

    model, optimizer, training_loss = utils.train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        x_train=x_train,
        y_train=y_train)

    accuracy, pred_proba, pred_class = utils.eval_model(
        model=model,
        x_test=x_test,
        y_test=y_test)

    utils.plot_all(
        model=model,
        training_loss=training_loss)


if __name__ == '__main__':
    args = utils.create_args()
    main(args)
