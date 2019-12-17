import os
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from functools import partial
import tqdm
import numpy as np
import plotly.graph_objects as go

import os, sys 
sys.path.append("../")
#debugging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import utils_run
from utils_data import create_data, create_data_mixture_of_gaussians
from utils_run import train_model, eval_model


class My_Net(nn.Module):
    """
    Network to experiments different configurations->easier. 
    """
    def __init__(self,
                 n_in,
                 n_out,
                 alpha_0=1.,
                 n_hidden=12, 
                 dropout_p=0., 
                 epsilon=0.1):

        super(My_Net, self).__init__()
        self.alpha_0 = alpha_0
        self.epsilon = epsilon #For overfflow
        self.feedforward_layers = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.Dropout(p=dropout_p),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_out))

    def forward(self, x):
        logits = self.feedforward_layers(x)
        alphas = torch.exp(logits)  + self.epsilon
        mean = alphas / alphas.sum(dim=1).unsqueeze(dim=1)
        precision = torch.sum(alphas, axis=1)
        return logits, mean, alphas, precision

def get_sharp_indices(target_conc_train, model_conc_train, sharp=101, flat=1):
    """
    Inputs:
        -target_conc_train: tensors: (total nb of batches in training, batch_size, nb_classes)
            The tensor of the target concentrations used for training the model. Obtained form train model.
        -model_conc_train: Similar.
    Outputs:
        -outputs: list, length=nb_classes.
                l[0] = tuple of size 2
            For each class, gives the corresponding points in the training data
            AND the output of the network for these same points.
    """
    nb_classes = target_conc_train.shape[-1]
    outputs = []
    for i in range(nb_classes):
        # The data points for which, class i.
        sharp_indices = (target_conc_train[:, :, i] == sharp)
        outputs.append((sharp_indices, model_conc_train[sharp_indices]))
    return outputs

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

# Create Data - No overlap, clear. 
shuffle = True 
data = create_data([
    create_data_mixture_of_gaussians],
    [mog_three_in_distribution], shuffle=shuffle)

#Add args for exp.
parser = utils_run.create_arg_parser()
parser.add_argument('--dropout_p', default=0.0, help='Add dropout')
parser.add_argument('--epsilon', default=0.1, help='Smoothing value')
parser.add_argument('--track_mode', action='store_false', 
help='Keep track of the training loss only or also the different output values.')
parser.add_argument('--verbose', action='store_true', 
help='If verbose, print loss values obtained during training')
parser.add_argument('--renderer', default='png', 
help='rendered for png/vs interactive plots. Choose png or chrome for instance')
args = parser.parse_args()


def main(args, data):
    #Define model and args
    model = My_Net(n_in=2, n_out=3, n_hidden=12, epsilon=args.epsilon, dropout_p=args.dropout_p) #Basic. 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.reverse:
        print('Using the reverse KL objective')
        loss_fn = utils_run.kl_backward
    else:
        print('Using the forward KL objective')
        loss_fn = utils_run.kl_forward
    #mode = 'reverse' if args.reverse else 'forward'
    # Train
    x_train, targets, target_concentrations = (data.values())
    print(f'Training for {args.n_epochs} epochs')
    model, optimizer, tracks = train_model(model,
                optimizer,
                loss_fn,
                args.n_epochs,
                args.batch_size,
                x_train,
                target_concentrations, 
                verbose=args.verbose,
                track_mode=args.track_mode)
    #Get all the info stored form the training              
    precision_train = torch.stack(tracks['precision'])
    model_conc_train = torch.stack(tracks['model_concentrations'])
    target_conc_train = torch.stack(tracks['target_concentrations'])
    out_train = get_sharp_indices(target_conc_train, model_conc_train)
    # Eval
    model.eval()
    _,_,model_conc_test,_ = model(x_train)
    out_eval = get_sharp_indices(target_concentrations.unsqueeze(1),
                             model_conc_test.unsqueeze(1))
    #=============
    # Plots
    #===========
    utils_run.plot_results(x_train,
                 targets,
                 target_concentrations,
                 model,
                tracks['training_loss'])
    utils_run.plot_alphas_train(out_train, nb_classes=3, renderer=args.renderer)
    utils_run.plot_alphas_histogram(
        out_eval, nb_classes=3, renderer=args.renderer)
    utils_run.plot_precision_train(precision_train, renderer=args.renderer)
  
    return model, tracks

if __name__ == "__main__":
    print('Loaded the experimental setup')
    #main(args, data)

