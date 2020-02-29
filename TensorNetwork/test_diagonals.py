
#################################################################
#                                                               #
#                   DIAGONALS CLASSIFIER                        #
#                                                               #
# Script used to test the trained network that performs a       #
# binary classification of a generated dataset of images        #
# containing one of the two diagonals of a square               #
# (with some added noise)                                       #
#                                                               #
# Run default script with:                                      #
#                                                               #
#           python test_diagonals.py                            #
#                                                               #
#           If you want to set some custom parameter,           #
#           add --arg = x to the command.                       #
#                                                               #
#################################################################

import numpy as np
import argparse
import pickle

import data_generator as gen
import Network_class as tn


########## PARAMETERS (set what you want) ############
parser = argparse.ArgumentParser(description='Train the Tensor Network to classify a binary MNIST dataset')

parser.add_argument('--filename', type=str, default='trained_diag_model.dat', help='Filename of the trained network')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples in the test dataset')
parser.add_argument('--sigma', type=float, default=0.6, help='Sigma of the noise that will be added to the dataset')

args = parser.parse_args()
#####################################################



# Load the trained network
with open(args.filename, 'rb') as file: 
    net = pickle.load(file) 

# Generate a test dataset
linear_dim = np.sqrt(net.N).astype('int')
(data, label) = gen.create_dataset(args.n_samples, linear_dim, args.sigma)
batch_size = {'train_batch_size':1, 'val_batch_size':1, 'test_batch_size':128}
_, _, test_loader = gen.prepare_dataset(data, label, 0, 0, **batch_size)


"""
# Compute prediction of the model over the generated dataset
data = next(iter(test_loader))
x = np.array([data[i][0] for i in range(len(data))])
y = np.array([data[i][1] for i in range(len(data))])
f = net.apply_act_func(net.forward(x))
"""


# For each batch in the dataset compute accuracy and MAE
batch_acc = np.zeros(len(test_loader))
batch_MAE = np.zeros(len(test_loader))

for i, data in enumerate(test_loader, 0):

    # Compute prediction of the model over the generated dataset
    x = np.array([data[i][0] for i in range(len(data))])
    y = np.array([data[i][1] for i in range(len(data))])
    f = net.apply_act_func(net.forward(x))

    # Compute accuracy
    batch_acc[i] = net.accuracy(x,y,f)

    # Compute MAE
    batch_MAE[i] = np.abs(y-f.elem).mean()

# Average results over batches and print results
print('\tAccuracy:            ', batch_acc.mean())
print('\tMean Absolute Error: ', batch_MAE.mean())