
#################################################################
#                                                               #
#                   DIAGONALS CLASSIFIER                        #
#                                                               #
# Script used to train the Tensor network to perform a binary   #
# classification of a generated dataset of images containing    #
# one of the two diagonals of a square (with some added noise)  #
#                                                               #
# Run default script with:                                      #
#                                                               #
#           python training_diagonals.py                        #
#                                                               #
#           If you want to set some custom parameter,           #
#           add --arg = x to the command.                       #
#                                                               #
#################################################################


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

import data_generator as gen
import Network_class as tn



########## PARAMETERS (set what you want) ############
parser = argparse.ArgumentParser(description='Train the Tensor Network to classify the dataset of diagonals')

parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples to generate (i.e. size of the dataset)')
parser.add_argument('--linear_dim', type=int, default=8, help='Size of both dimensions of the samples')
parser.add_argument('--sigma', type=float, default=0.7, help='Sigma of the noise that will be added to the dataset')
parser.add_argument('--n_train_batch', type=int, default=1, help='Number of batches in which the training set will be split')

parser.add_argument('--M', type=int, default=10, help='Size of the bond between tensors of the network')
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--L2_decay', type=float, default=1, help='Weight decay value for L2 regularization')

parser.add_argument('--act_fn', type=str, default='softmax', help="Activation function of the output (can be 'linear', 'sigmoid' or 'softmax')")
parser.add_argument('--loss_fn', type=str, default='full_cross_ent', help="Loss function (can be 'MSE', 'cross_entropy' or 'full_cross_ent')")

args = parser.parse_args()

#####################################################



# Split dataset in train and validation and build the data_loaders
train_batch = int( args.n_samples * (0.8) / args.n_train_batch)
(data, label) = gen.create_dataset(args.n_samples, args.linear_dim, args.sigma)
batch_size = {'train_batch_size':train_batch, 'val_batch_size':128, 'test_batch_size':128}
train_loader, val_loader, test_loader = gen.prepare_dataset(data, label, 1, 0.2, **batch_size)

# Build the Tensor Network
calibration_batch = next(iter(train_loader))
x_calibration = np.array([calibration_batch[i][0] for i in range(len(calibration_batch))])
net = tn.Network(N=args.linear_dim**2, M=args.M, L=2, calibration_X=x_calibration, 
                 normalize=True, act_fn=args.act_fn, loss_fn=args.loss_fn)

# Train the network
val_acc, var_hist = net.train(train_loader, val_loader, lr=args.lr, 
                              n_epochs=args.n_epochs, weight_dec=args.L2_decay)

# Save trained model
with open('trained_diag_model.dat', 'wb') as file:
    pickle.dump(net, file)



##### Plot the results #####
x_values = np.arange( args.n_epochs * var_hist.shape[2] ) / var_hist.shape[2] 

# Accuracies
plt.plot(x_values, var_hist[:,0].reshape(-1), label="Train acc")
plt.plot(np.arange(1,6), val_acc, 'ro', label='Validation acc')

plt.title("Accuracies of the network")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig('results/diag_accuracy.png')
plt.close()

# Mean Absolute Error
plt.plot(x_values, var_hist[:,1].reshape(-1), label='MAE')
plt.title("Mean Absolute Error")
plt.ylabel("| f(x) - y |")
plt.xlabel("Epoch")
plt.legend()
plt.savefig('results/diag_MAE.png')

print("\nPlots are stored in the 'results' folder\n")