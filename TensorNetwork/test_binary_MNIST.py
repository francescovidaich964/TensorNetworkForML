
#################################################################
#                                                               #
#                   BINARY MNIST CLASSIFIER	                    #
#                                                               #
# Script used to test the trained network that performs a       #
# binary classification of the (reduced) MNIST dataset.         #
#                                                               #
# Run default script with:                                      #
#                                                               #
#           python test_binary_MNIST.py                         #
#                                                               #
#           If you want to set some custom parameter,           #
#           add --arg = x to the command.                       #
#                                                               #
#################################################################

import numpy as np
import argparse
import pickle
import skimage.measure

import data_generator as gen
import Network_class as tn



# Function that reduces the resolution of MNIST images
def pooling(X):
    X = skimage.measure.block_reduce(X, (1,2,2), np.max)
    return X


########## PARAMETERS (set what you want) ############
parser = argparse.ArgumentParser(description='Train the Tensor Network to classify a binary MNIST dataset')

parser.add_argument('--filename', type=str, default='trained_MNIST_model.dat', help='Filename of the trained network')
parser.add_argument('--data_dir', type=str, default='datasets', help='Directory where the MNIST dataset is stored')

args = parser.parse_args()
#####################################################



# Load the trained network
with open(args.filename, 'rb') as file: 
    net = pickle.load(file) 


# Load the MNIST full dataset and reduce its resolution
MNIST_data = gen.get_MNIST_dataset(data_root_dir = args.data_dir)
train_data, train_labels, test_data, test_labels = MNIST_data
data = np.concatenate((train_data,test_data))
data = pooling(data)
labels = np.concatenate((train_labels,test_labels))

# Extract only 0s and 1s images from the dataset
mask1 = (labels == 0) 
mask2 = (labels == 1)
mask = mask1 + mask2
labels01 = labels[mask]
data01 = data[mask]

# Generate a test dataset
batch_size = {'train_batch_size':1, 'val_batch_size':1, 'test_batch_size':128}
_, _, test_loader = gen.prepare_dataset(data01, labels01, 0, 0, **batch_size)

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
