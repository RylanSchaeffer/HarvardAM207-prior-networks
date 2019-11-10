

#%%
import torch.nn.functional as F
import tqdm 
import numpy as np
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                   TensorDataset)

from torch.distributions import Dirichlet
from torch.distributions import kl_divergence
from torch.distributions.kl import _kl_dirichlet_dirichlet 
import argparse
#Analytical derivation with the digamma and so on. Just need the concentration parameters actually. 

#%%
class Toy_Net(torch.nn.Module):
    def __init__(self, n_in, n_out, alpha_0=1. , n_hidden=12):
        super(Toy_Net, self).__init__()
        self.alpha_0 = alpha_0
        self.linear_block = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_out),
            #nn.Sigmoid()
        )
    def forward(self, x):
        logits = self.linear_block(x)
        mean = F.softmax(logits / self.alpha_0, dim=1) #I have doubts about alpha 0..
        alphas = torch.exp(logits)
        precision = torch.sum(alphas)
        return logits, mean, alphas, precision

class Toy_LogReg(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(Toy_LogReg, self).__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x):
            return self.linear(x)

#Rylan's Net-returns a softmax of the alphas weighted by the alpha0. 
#Do we want the alpha 0 here ? 
class Network(nn.Module):
    def __init__(self,
                 alpha_0=3.):
        super().__init__()
        self.alpha_0 = alpha_0
        self.weights = nn.Linear(
            in_features=2,
            out_features=3)
    def forward(self, x):
        alphas = self.weights(x)
        assert_no_nan_no_inf(alphas)
        output = F.softmax(alphas / self.alpha_0, dim=1)
        assert_no_nan_no_inf(output)
        return output

def train(model, train_loader, optimizer, n_epochs=10):
    loss_trace = []
    for epoch in range(n_epochs):
        for batch in tqdm.tqdm(train_loader, desc=f'Training for epoch {epoch}'):
            optimizer.zero_grad() #clean the graph
            x, y = batch
            y_hat, mean, alphas, precision = model(x)
            loss = loss_fn(y_hat, y.squeeze())
            loss_trace.append(loss)
            loss.backward() #backprop
            optimizer.step() #step
        print("Last obtained loss", loss)
    return model, optimizer, loss_trace #for post-training

def evaluate(model, x_test, y_test, test_loader=None):
    #TODO-Add a sequenial test loader
    y_hat, _, _,_ = model(x_test)
    pred_proba, pred_class = torch.max(y_hat, 1)
    accuracy = (pred_class == y_test.squeeze()).sum()/len(y_test)
    return accuracy, pred_proba, pred_class

def generate_dataset(points_per_cluster=100, nb_clusters=3, scale=10, print=True, *args):
    """
    #TODO-implement so that we can actually choose other inputs
    - Args to specify the mean variances if need be, under the form mu_1, sigma_1" and so on
    output: np.array with samples from each gaussian.
    """
    mu_1, sigma_1 = scale * \
        np.array([0.0, 1.0]), np.array([[2.0, 0], [0, 2.0]])
    mu_2, sigma_2 = scale * \
        np.array([-np.sqrt(3)/2, -1. / 2]), np.array([[2.0, 0], [0, 2.0]])
    mu_3, sigma_3 = scale * \
        np.array([np.sqrt(3)/2, -1./2]), np.array([[2.0, 0], [0, 2.0]])

    X_1 = np.random.multivariate_normal(
        mean=mu_1, cov=sigma_1, size=points_per_cluster)
    X_2 = np.random.multivariate_normal(
        mean=mu_2, cov=sigma_2, size=points_per_cluster)
    X_3 = np.random.multivariate_normal(
        mean=mu_3, cov=sigma_3, size=points_per_cluster)

    Y_1, Y_2, Y_3 = np.zeros((len(X_1), 1)), np.ones(
        (len(X_1), 1)), 2*np.ones((len(X_1), 1))
    if print:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        ax.scatter(X_1[:, 0], X_1[:, 1], color='r')
        ax.scatter(X_2[:, 0], X_2[:, 1], color='b')
        ax.scatter(X_3[:, 0], X_3[:, 1], color='y')
        plt.show()
    return X_1, Y_1, X_2, Y_2, X_3, Y_3

def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def get_target_dirichlet(y_train, alpha_0, nb_class=3, epsilon=1e-4):
    """alpha_0  Hyperparameter-> specify the sharpness.
    epsilon ->Smoothing param
    nb_class 
    output: target_dirichlet -> torch.distributions.Dirichlet. 
    access to concentration parameters with output.concentration"""
    one_hot_labels = F.one_hot(y_train.squeeze()).to(torch.float32)
    soft_labels = one_hot_labels - one_hot_labels * nb_class * epsilon + epsilon
    target_concentrations = alpha_0 * soft_labels
    #target_dirichlet = Dirichlet(target_concentrations)
    return target_concentrations


def kl_loss(model_softmax_outputs, target_concentrations, reverse=True):
    """
    Input: Model softmax outputs or anything else that we want to build
    our Dirichlet distribution on.
    """
    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(
        model_softmax_outputs)  # is that what we want ? Our concentrations parameters. sum to one at the end, no matter what alpha_0
    if reverse:
        kl_divs = _kl_dirichlet_dirichlet(
            p=target_dirichlet, q=model_dirichlet)
    else:  # forward
        kl_divs = _kl_dirichlet_dirichlet(
            p=model_dirichlet, q=target_dirichlet)
    assert_no_nan_no_inf(kl_divs)
    mean_kl = torch.mean(kl_divs)
    return mean_kl

def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse', help='whether to use forward or backward kl div', 
    action='store_true')
    parser.add_argument('--n_epochs', help='nb of epochs to train', type=int, default=10)
    parser.add_argument('--scale', help="Scale of the synt. data", type=float,
    default=10.0)
    parser.add_argument('--lr', help="learning rate", type=float,
                       default=0.01)

    return parser


"""
If you are using nn.CrossentropyLoss, you should pass the logits without any non-linearity to the criterion, as internally F.log_softmax and nn.NLLLoss will be used.
Alternatively, you could apply F.log_softmax manually and pass it directly to nn.NLLLoss.

So usually you don’t use a softmax activation, unless e.g. you are implementing some custom criterion of course.
"""
#%%
#TODO: #DO NOT OAY ATTENTION TO THE FOLLOWING
#Define target means
#Implement Forward KL.
# Implement loss_fn as forward KL. Using the loss.KL Divergence of torch so that we dont have to specify everything ? 
# If we want to use torch.distributions.kl, we need two Dirichlet distributions per se --> we can estimate the parameters of the empirical distribution 
#####First approach = from the output of the network, estimate the corresponding empirical dirichlet -> in that case, we have one approximation per batch...
# Second approach, directly use the torch.nn.KLDivLoss() -> equivalent to F.kl but using the empirical data only. 
# The KL Divergence will be between: 

# we want to model p(\mu | x, \theta) =Dir(u, f(x,\theta)) -> We need the outpu of the parameter to be all positive. How do we do that ? What is the ast layer of the network then ?
# The theoertical tagte is know--> np pb. We can then define a torch.distrib.kl_divergence or a F.kl directly. 
# We do not want the softmax output otherwise we will lose the sharpness part of the parameters. 
#%%
"""
notebook=True
if notebook: 
    #########################PARAMETERS####################
    batch_size = 32
    lr = 0.01
    loss_fn = torch.nn.CrossEntropyLoss()
    n_epochs = 200
    #######################################################

    device = "gpu:0" if print(torch.cuda.is_available()) else "cpu"
    device = torch.device(device)
    print("Working on device-", device)
    X_1, Y_1, X_2, Y_2, X_3, Y_3 = generate_dataset(print=True)

    #Construct simple tensors.
    X_train = np.concatenate([X_1, X_2, X_3], axis=0)
    Y_train = np.concatenate([Y_1, Y_2, Y_3], axis=0)
    x_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(Y_train, dtype=torch.long)

    target_dirichlet = get_target_dirichlet(y_train, alpha_0=10)
    print(target_dirichlet.concentration.size())
    my_net = Toy_Net(2,3)
    logits, mean, alphas, precision = my_net(x_train)
    
    print(torch.mean(_kl_dirichlet_dirichlet(target_dirichlet, Dirichlet(alphas))))
    print(torch.mean(_kl_dirichlet_dirichlet(Dirichlet(alphas), target_dirichlet)))
    try:
        l = kl_loss(alphas, target_dirichlet)
        print(l)
    except Exception as e: 
        print(e)
"""

#%%
if __name__=='__main__':
    #########################PARAMETERS####################
    parser = my_parser()
    args = parser.parse_args()
    batch_size = 32
    lr = 0.01
    loss_fn = torch.nn.CrossEntropyLoss()
    n_epochs = args.n_epochs
    #######################################################
    
    device = "gpu:0" if print(torch.cuda.is_available()) else "cpu"
    device = torch.device(device)
    print("Working on device-", device)
    X_1, Y_1, X_2, Y_2, X_3, Y_3 = generate_dataset(print=True, scale=args.scale)
    
    #Construct tensors.
    X_train = np.concatenate([X_1, X_2, X_3], axis=0)
    Y_train = np.concatenate([Y_1, Y_2, Y_3], axis=0)
    x_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(Y_train, dtype=torch.long) #type long for the target. 

    #Define dataset, sampler, loader
    train_dataset = TensorDataset(x_train, y_train)
    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, sampler=sampler, batch_size=batch_size)
    
    #Define model, optimizer
    model = Toy_Net(n_in=2,n_out=3,n_hidden=12)
    #model = Toy_LogReg(2,3)
    for name, param in model.named_parameters():
        print(name, "Requires grad?", param.requires_grad)
    optimizer = torch.optim.Adam(model.parameters())
    #Train and evaluate post-training
    model, optimizer, loss_trace = train(model, train_loader, optimizer, n_epochs=2)
    acc ,_ , _ = evaluate(model, x_train, y_train) #Overfits relatively easy.
    print("Accuracy on the training set-", acc.to(torch.float64).item())
    plt.plot(loss_trace)
    plt.show()

    #######SECOND TRAINING WITH THE FIRST KL loss####################
    target_concentrations = get_target_dirichlet(
        y_train, alpha_0=2)  # new 'labels'
    train_dataset = TensorDataset(x_train, target_concentrations)
    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, sampler=sampler, batch_size=batch_size)
    
    model = Toy_Net(n_in=2, n_out=3, n_hidden=12) #Reinstantiate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_trace = []
    for epoch in range(n_epochs):
        for batch in tqdm.tqdm(train_loader, desc=f'Training for epoch {epoch}'):
            optimizer.zero_grad()  # clean the graph
            x, target_conc = batch
            y_hat, mean, alphas, precision = model(x)
            #loss = loss_fn(y_hat, y.squeeze())
            loss = kl_loss(alphas, target_conc, reverse=args.reverse)
            loss_trace.append(loss)
            loss.backward()  # backprop
            optimizer.step()  # step
        print("Last obtained loss", loss)
    plt.plot(loss_trace[50:])
    plt.show()
    ########EVALUATE###########
    model.eval()
    with torch.no_grad():
        y_hat, mean, alphas, precision = model(x_train)
        pred_proba, pred_class = torch.max(y_hat, dim=1)
    print("Acc", (pred_class == y_train.squeeze()).sum() .item()/len(y_train))
    fig, ax =plt.subplots(1,3,figsize=(20,20))
    ax[0].scatter(x_train[y_train.squeeze() == 0, 0],
               x_train[y_train.squeeze() == 0, 1], marker='x', color='r')
    ax[0].scatter(x_train[y_train.squeeze() == 1, 0],
               x_train[y_train.squeeze() == 1, 1], marker='x', color='b')
    ax[0].scatter(x_train[y_train.squeeze() == 2, 0],
               x_train[y_train.squeeze() == 2, 1], marker='+', color='y')

    ax[0].scatter(x_train[pred_class == 0, 0], x_train[pred_class==0, 1], color='r', alpha=0.3)
    ax[0].scatter(x_train[pred_class == 1, 0], x_train[pred_class == 1, 1],color='b', alpha=0.3)
    ax[0].scatter(x_train[pred_class == 2, 0], x_train[pred_class == 2, 1],color='y', alpha=0.3)
    
    ax[1].scatter(x_train[y_train.squeeze() == 0, 0],
                  x_train[y_train.squeeze() == 0, 1], marker='x', color='r')
    ax[1].scatter(x_train[y_train.squeeze() == 1, 0],
                  x_train[y_train.squeeze() == 1, 1], marker='x', color='b')
    ax[1].scatter(x_train[y_train.squeeze() == 2, 0],
                  x_train[y_train.squeeze() == 2, 1], marker='+', color='y')
    
    ax[2].scatter(x_train[pred_class == 0, 0],
                  x_train[pred_class == 0, 1], color='r', alpha=0.3)
    ax[2].scatter(x_train[pred_class == 1, 0],
                  x_train[pred_class == 1, 1], color='b', alpha=0.3)
    ax[2].scatter(x_train[pred_class == 2, 0],
                  x_train[pred_class == 2, 1], color='y', alpha=0.3)
    
    
    plt.show()



    
    


