import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, TensorDataset, SubsetRandomSampler


def create_dataset(n_samples, linear_dim=5, sigma=0.5, prob_zero=0.5):
    """
    Create a dataset of greyscale images with 2 different patterns and their labels.
    
    Parameters
    ----------
    n_samples : int, 
        number of samples to be created
    linear_dim : int, 
        linear dimension of squared images
    sigma : float in [0,1], 
        level of noise (0 = no noise, 1 = only noise)
    prob_zero : float in [0,1], 
        probability that an image is created according to pattern 0
        
    Returns
    -------
    data : numpy array of floats in [0,1] of shape (n_samples, linear_dim, linear_dim)
        array of greyscale images
    labels : numpy array of int of length n_samples
        labels of the corresponding images (either 0 or 1)
    
    Notes
    -----
    Example of use case:
    
    import data_generator as gen
    n_samples = 10 
    data, labels = gen.create_dataset(n_samples)
    
    """
    # true images (names=labels)
    one = np.eye(linear_dim)
    zero = one[::-1,:]
    
    # sample labels according to the prob of having a label = 0 (prob_zero)
    labels = np.random.choice([0,1], size=n_samples, p=[prob_zero, 1-prob_zero])
    
    data = np.zeros((n_samples,linear_dim,linear_dim))
    zero_mask = (labels==0)
    data[zero_mask] = zero
    data[~zero_mask] = one
    # add noise
    noise = np.random.rand(n_samples,linear_dim,linear_dim)*sigma
    data = data*(1-sigma)+noise
    
    return data, labels


def get_MNIST_dataset(data_root_dir = './datasets', download=True):
    """
    Import MNIST dataset in numpy splitted in training and test sets.
    
    Parameters
    ----------
    data_root_dir : str,
        path of the directory where to download or import the dataset
    download: bool,
        downloads the dataset from http://yann.lecun.com/exdb/mnist/
        
    Return
    ------
    train_data : numpy array, float, shape (60000, 28, 28)
    train_labels : numpy array, int, shape (60000,)
    test_data : numpy array, float, shape (10000, 28, 28)
    test_labels : numpy array, int, shape (10000,)
    
    Notes
    -----
    Requires torchvision library

    """
    torch_train_dataset = MNIST(data_root_dir, train=True,  download=download)
    torch_test_dataset  = MNIST(data_root_dir, train=False, download=download)
    
    train_data = np.array([np.array(x[0]) for x in torch_train_dataset])
    train_labels = np.array([np.array(x[1]) for x in torch_train_dataset])

    test_data = np.array([np.array(x[0]) for x in torch_test_dataset])
    test_labels = np.array([np.array(x[1]) for x in torch_test_dataset])
    
    return train_data, train_labels, test_data, test_labels


class NumpyDataset(Dataset):
    """
    Class inherited from torch.Dataset; it can contain any dataset 
    made of numpy samples in order to make it compatible with 
    torch.Dataloader to split it in batches.
    
    Attributes
    ----------
    data: numpy array
        Input images of the dataset
    label: numpy array
        Labels of the input images

    Methods
    -------
    __init__(self, data, label)
        Build instance of a NumpyDataset with the input dataset
    __len__(self)
        Retunr the number of samples contained in the dataset
    __getitem__(self, index)
        Return the sample (x,y) corresponding to the selected index

    """
    
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (self.data[index], self.label[index])


def prepare_dataset(data, label, train_perc, val_perc, train_batch_size, val_batch_size, test_batch_size):
    """
    Encode input as a mixed state of 0 and 1, split and prepare dataloaders
    for train, validation and test datasets 
    
    Parameters
    ----------
    data: numpy array,
        Images of the input dataset 
    label: bool,
        Labels of the input images
    train_perc: float
        Percentage of splitting between train and test dataset
        (e.g. 1-train_perc %  of the dataset will be used for test)
    val_perc: float
        Percentage of the training dataset that will be used as validation
    train_batch_size: int
        Batch size of the training dataset
    validation_batch_size: int
        Batch size of the validation dataset
    test_batch_size: int
        Batch size of the test dataset

    Return
    ------
    train_loader:
        Dataloader of the training set
    val_loader:
        Dataloader of the validation set
    test_loader:
        Dataloader of the test set
    
    Notes
    -----
    Requires torch.Dataset and torch.Dataloader

    """


    
    def psi(x):
        x = np.array((np.sin(np.pi*x/2),np.cos(np.pi*x/2)))
        return np.transpose(x, [1,2,0])


    # flatten images
    x = data.reshape(len(data),-1)
    # embedd them
    x = psi(x)
    
    # training/test splitting
    m = int(len(x)*train_perc)
    x_train= x[:m]
    y_train = label[:m]
    x_test =  x[m:]
    y_test = label[m:]
    
    # define custom NumpyDatasets
    train_set = NumpyDataset(x_train, y_train)
    test_set =  NumpyDataset(x_test, y_test)
   
    train_len = int(m*(1-val_perc))
    train_sampler = SubsetRandomSampler(np.arange(train_len))
    val_sampler = SubsetRandomSampler(np.arange(train_len,m))

    train_loader = DataLoader(train_set, train_batch_size, sampler=train_sampler, drop_last=True, collate_fn=lambda x: x)
    val_loader = DataLoader(train_set, val_batch_size, sampler=val_sampler, drop_last=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_set, test_batch_size, drop_last=False, collate_fn=lambda x: x)

    return train_loader, val_loader, test_loader