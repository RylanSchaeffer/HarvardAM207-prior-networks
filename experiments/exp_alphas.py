import torch 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os, sys 
sys.path.append("../")
#debugging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import utils_exp #Old for plot_results. #TODO: Remove and debug utils.plt
from utils.data import create_data, create_data_mixture_of_gaussians
from utils.models import ExperimentalNetwork
from utils import run, plot, measures


# Create Data - No overlap, clear.
#shuffle = True


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

data = create_data([
    create_data_mixture_of_gaussians],
    [mog_three_in_distribution])

def experimental_setup():
    parser = run.create_arg_parser()
    parser.add_argument('--dropout_p', default=0.0, help='Add dropout')
    parser.add_argument('--epsilon', default=0.1, help='Smoothing value')
    parser.add_argument('--verbose', action='store_true',
                    help='If verbose, print loss values obtained during training')
    parser.add_argument('--renderer', default='png',
                    help='rendered for png/vs interactive plots. Choose png or chrome for instance')
    args = parser.parse_args()
    args = vars(args)
    #Turn into dict
    if not args['track_mode']:
        args['track_mode'] = True
    #'Reverse is always true unless we set it manually to false.'
    args['mode'] = 'reverse' if args['reverse'] else 'forward'
    print('Loaded the experimental setup')
    return args
    
def main(args, data):
    #Model, optim, loss_fn
    model = ExperimentalNetwork(2, 3,[12], epsilon=args['epsilon'] )
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = measures.kl_loss_fn
    # Train
    print('Training for {} epochs, using {}'.format(
        args['n_epochs'], args['mode']))
    model, optimizer, tracks = run.train_model_exp(model,
                optimizer,
                loss_fn,
                args['n_epochs'],
                args['batch_size'],
                data,
                args)
    #Get all the info stored form the training
    #TODO-Turn into function             
    precision_train = torch.stack(tracks['precision'])
    model_conc_train = torch.stack(tracks['concentrations'])
    target_conc_train = torch.stack(tracks['y_concentrations_batch'])
    out_train = plot.get_sharp_indices(target_conc_train, model_conc_train)
    # Eval
    model.eval()
    x_train, targets, target_concentrations = (data.values())
    
    model_outputs = model(x_train)
    model_conc_test = model_outputs['concentrations']
    out_eval = plot.get_sharp_indices(target_concentrations.unsqueeze(1),
                             model_conc_test.unsqueeze(1))
    #=============
    # Plots
    #===========
    #TODO--> Debug the new plot_results
    utils_exp.plot_results(x_train,
                 targets,
                 target_concentrations,
                 model,
                tracks['training_loss'])
    plot.plot_alphas_train(out_train, nb_classes=3, renderer=args['renderer'])
    plot.plot_alphas_histogram(
        out_eval, nb_classes=3, renderer=args['renderer'])
    plot.plot_precision_train(precision_train, renderer=args['renderer'])
    return model, tracks

if __name__ == "__main__":
    args = experimental_setup()
    #main(args, data)

