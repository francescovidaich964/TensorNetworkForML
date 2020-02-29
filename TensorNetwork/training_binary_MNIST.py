
#################################################################
#                                                               #
#                   BINARY MNIST CLASSIFIER                     #
#                                                               #
# Script used to train the Tensor network to perform a binary   #
# classification of the (reduced) MNIST dataset.                #
#                                                               #
# Run default script with:                                      #
#                                                               #
#           python training_binary_MNIST.py                     #
#                                                               #
#           If you want to set some custom parameter,           #
#           add --arg = x to the command.                       #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
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

parser.add_argument('--data_dir', type=str, default='datasets', help='Directory where the MNIST dataset is stored')
parser.add_argument('--n_train_batch', type=int, default=10, help='Number of batches in which the training set will be split')
parser.add_argument('--M', type=int, default=3, help='Size of the bond between tensors of the network')
parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--L2_decay', type=float, default=1e-56, help='Weight decay value for L2 regularization')

parser.add_argument('--act_fn', type=str, default='softmax', help="Activation function of the output (can be 'linear', 'sigmoid' or 'softmax')")
parser.add_argument('--loss_fn', type=str, default='full_cross_ent', help="Loss function (can be 'MSE', 'cross_entropy' or 'full_cross_ent')")

args = parser.parse_args()

#####################################################


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

# Split dataset in train and validation and build the data_loaders
train_batch = int( len(data01) * (0.8) / args.n_train_batch)
batch_size = {'train_batch_size':train_batch, 'val_batch_size':128, 'test_batch_size':128}
train_loader, val_loader, test_loader = gen.prepare_dataset(data01, labels01, 1, 0.2, **batch_size)

# Build the Tensor Network
calibration_batch = next(iter(train_loader))
x_calibration = np.array([calibration_batch[i][0] for i in range(len(calibration_batch))])
net = tn.Network(N=data[0].size, M=args.M, L=2, calibration_X=x_calibration, 
                 normalize=True, act_fn=args.act_fn, loss_fn=args.loss_fn)

# Train the network
val_acc, var_hist = net.train(train_loader, val_loader, lr=args.lr, 
                              n_epochs=args.n_epochs, weight_dec=args.L2_decay)

# Save trained model
with open('trained_MNIST_model.dat', 'wb') as file:
    pickle.dump(net, file)




##### Plot the results #####
x_values = np.arange( args.n_epochs * var_hist.shape[2] ) / var_hist.shape[2] 

# Accuracies
plt.plot(x_values, var_hist[:,0].reshape(-1), label="Train acc")
plt.plot(np.arange(1,args.n_epochs+1), val_acc, 'ro', label='Validation acc')

plt.title("Accuracies of the network")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig('results/MNIST_accuracy.png')
plt.close()

# Mean Absolute Error
plt.plot(x_values, var_hist[:,1].reshape(-1), label='MAE')
plt.title("Mean Absolute Error")
plt.ylabel("| f(x) - y |")
plt.xlabel("Epoch")
plt.legend()
plt.savefig('results/MNIST_MAE.png')
plt.close()

print("\nPlots are stored in the 'results' folder\n")
