
import numpy as np
import copy
from tqdm import tnrange
new = np.newaxis

from Tensor_class import Tensor
from custom_linalg_tools import contract, partial_trace

class Network():
    """ 
    Class used to represent a Matrix Product State (MPS) Tensor Network.
    
    Attributes
    ----------
    N: int
        First dimension of the input (e.g. number of pixels for an image)
    D: int
        Second dimension of the input (e.g. number of color channels or
        embedding dimension)
    L: int
        Number of possible labels
    M: int
        Initial bond dimension
    normalize: bool 
        If True, normalizes the weights so that the expected output is of order 1
        for inputs X with all entries in [0,1]   
    As: list of Tensors
        List of matrix-like tensors of the MPS network
    l_pos: int
        Index of As at which there is the Tensor with the label dimension 
    TX: list of Tensors
        Object to wrap the input as a list of Tensors in order to perform
        contractions with the As
    r_cum_contraction: list of Tensors
        List of cumulative contractions from the "right" to the "left" of the network
        Calling A_TX the list of the contracted tensors As[i] TX[i], the first element 
        of cum_contraction is A_TX[-1], then the second is the contraction of A_TX[-1]
        and A_TX[-2], the third the contraction of cum_contraction[1] with A_TX[-3] 
        and so on
    l_cum_contraction: list of Tensor
        List of cumulative contractions from the "left" to the "right" of the network
        
    Methods
    -------
    forward(X)
        Computes predictions for input X
    train(train_loader, val_loader, lr, n_epochs = 10, early_stopping = False, print_freq = 100)
        Trains the network. 
    accuracy(x, y, f=None)
        Computes the accuracy of the networks' predictions
        
    Notes
    -----
    CHANGE IT - This class implements a circular MPS Tensor Network, meaning that the network topology is a ring
    where all the tensors are contracted both to the left and to the right.
    """
    
    def __init__(self, N, M, D=2, L=10, normalize=False, calibration_X=None):
        """
        Parameters
        ----------
        N: int
            First dimension of the input (e.g. number of pixels for an image)
        D: int
            Second dimension of the input (e.g. number of color channels or
            embedding dimension)
        L: int
            Number of possible labels
        M: int
            Initial bond dimension
        normalize: bool 
            If True, normalizes the weights so that the expected output is of order 1
            for inputs X with all entries in [0,1]
        calibration_X: numpy array
            Sample of shape (batch_size, N, D) used to calibrate the weights of the network before training
        """
        
        self.N = N
        self.D = D
        self.L = L
        self.M = M
        
        self.As = []
        
        if normalize:
            print('Normalizing weights...')
            # output goes like [M E(A) E(x) D]^N
            # E(A) expected value of entry A (tensor) distributed uniformly in [0,1] -> 0.5
            # E(x) expected value of x = cos(pi/2 * u) (or sin(pi/2 * u)), u in [0,1] -> 0.64
            scale = float(self.M)*0.5*0.64*self.D
            print('Scaling factor: %.2f'%scale)
            
            ### ORIGINAL ###
            #self.As.append(Tensor(shape=[L,M,M,D], axes_names=['l','left','right','d0'], scale = scale))
            #for i in range(1,N):
            #    self.As.append(Tensor(shape=[M,M,D], axes_names=['left','right','d'+str(i)], scale = scale))
            self.As.append(Tensor(shape=[L,M,D], axes_names=['l','right','d0'], scale = scale))
            for i in range(1,N-1):
                self.As.append(Tensor(shape=[M,M,D], axes_names=['left','right','d'+str(i)], scale = scale))
            self.As.append(Tensor(shape=[M,D], axes_names=['left','d'+str(N-1)], scale = scale))
            
            if calibration_X is None:
                # if calibration input not provided, create one from scratch
                def psi(x):
                    """Embedding function """
                    x = np.array((np.sin(np.pi*x/2),np.cos(np.pi*x/2)))
                    return np.transpose(x, [1,2,0])
                
                B = 16 # batch size of the samples used for calibration
                X = np.random.random((B, self.N))
                X = psi(X)
                
            else:
                X = calibration_X
                B = X.shape[0]
            
            print('Calibrating weights on dataset...')

            # compute the order of magnitude of the output
            f = self.forward(X)
            f_max = np.abs(f.elem).max().astype('float')
            print('f_max for random input of %d samples : '%(B),f_max)
            F2 = f_max**(1./self.N) # factor for rescaling
            print("Rescaling factor for calibration: ", F2)
            for i in range(self.N):
                self.As[i].elem = self.As[i].elem/F2
                
            # compute the new order of magnitude of the output (should be 1)
            f = self.forward(X)  
            f_max = np.abs(f.elem).max().astype('float')
            print('f_max for random input of %d samples (after): '%(B),f_max)
            
        else:
            # without normalization each entry of each A is a random number in [0,1]
            self.As.append(Tensor(shape=[L,M,M,D], axes_names=['l','left','right','d0']))
            for i in range(1,N):
                self.As.append(Tensor(shape=[M,M,D], axes_names=['left','right','d'+str(i)]))
        
        # position of the tensor with additional dimension for the output of the net
        self.l_pos = 0

        return
    

    def forward(self, X):
        """
        Computes predictions for input X
        
        Parameters
        ----------
        X : numpy array
            Shape (batch_size, N, D) (see init parameters for better explanation)
            
        Returns
        -------
        out: Tensor
            Tensor of shape (batch_size, L), where L is the number of possible labels
            out.elem shows the scores assigned by the network to each label
            (the argmax along the first axis yields the predicted label for each sample
            in the batch)
            
        Raises
        ------
        AssertionError
            If X.shape[1] does not match with self.N
        """
        
        assert self.N == X.shape[1], "The 1 dimension of the input data must be the flattened number of pixels"

        # X should be batch_size x 784 x 2
        TX = []
        for i in range(self.N):
            TX.append(Tensor(elem=X[:,i,:], axes_names=['b','d'+str(i)]))
                      
        # This must be futher investigate, three ways:
        #     * numpy vectorize
        #     * list comprehension
        #     * multithread
                      
        #A_TX = np.vectorize(contract)(TX, A, contracted='d'+str(i))
        
        A_TX = [contract(self.As[i], TX[i], contracted='d'+str(i)) for i in range(self.N)]
        cum_contraction = []
        cum_contraction.append(A_TX[-1])
        for j in range(1,self.N): 
            tmp_cum = copy.deepcopy(cum_contraction[-1])
            tmp_cum = contract(A_TX[-(j+1)], tmp_cum, 'right', 'left', common='b')
            cum_contraction.append(tmp_cum)

        self.r_cum_contraction = cum_contraction[::-1]
        self.left_contraction = None
        self.TX = TX

        ### ORIGINAL ###
        #out = partial_trace(self.r_cum_contraction[0], 'right', 'left') # close the circle
        #return out
        return self.r_cum_contraction[0]


    def train(self, train_loader, val_loader, lr, n_epochs = 10, early_stopping = False, print_freq = 100):
        """
        Trains the network
        
        Parameters
        ----------
        train_loader, val_loader: DataLoader (from torch.utils.data)
            Data loaders for training and validation set
            Must yield numpy arrays of shape (batch_size, N, D)
        lr: float in (0,1]
            Learning rate that multiplies the gradient
        n_epochs: int, optional
            Number of epochs of training (default=10). One epoch consists in a full
            sweep for each batch in the train_loader.
        early_stopping: bool, optional
            Not implemented. Default is False.
        print_freq: int, optional
            Number of batches after which prints a training update
            
        Returns
        -------
        train_acc, val_acc: list of float
            List of accuracies on training and validation sets for each epoch
        """
        
        train_acc = []
        val_acc = []

        # if early_stopping = False
        for epoch in tnrange(n_epochs, desc="Epoch loop", leave = True):
            epoch_train_acc = np.zeros(len(train_loader))
           
            # train
            print_every = int(len(train_loader)/print_freq)
            for i, data in enumerate(train_loader, 0):
                x = np.array([data[i][0] for i in range(len(data))])
                y = np.array([data[i][1] for i in range(len(data))])
                
                f = self.forward(x)
                batch_acc = self.accuracy(x, y, f) # compute accuracy before batch optimization

                ### ORIGINAL ###
                #f = self.sweep(x, y, f, lr)
                
                # Select r/l sweep depending on l_pos
                if self.l_pos == 0:
                    f = self.r_sweep(x, y, f, lr)
                elif self.l_pos == self.N-1:
                    f = self.l_sweep(x, y, f, lr)
                else:
                    print("\nl index is neither at the end or at the beginning of the net\n")
                
                
                ##################################################################################
                #batch_acc_opt = self.accuracy(x, y, f) # compute accuracy after batch optimization
                #print('\nbatch_acc: ', batch_acc)
                #print('batch_acc_opt: ', batch_acc_opt)
                ##################################################################################
                epoch_train_acc[i] = batch_acc
                
                #if (i+1) % (print_every) == 0:
                print('\r'+"Epoch %d - train accuracy : %.4f - completed : %.2f "%(epoch, epoch_train_acc[i], (i+1)*100/len(train_loader))+'%', end=' ')
                    
            train_acc.append(epoch_train_acc.mean())
            
            # validation
            epoch_val_acc = np.zeros(len(val_loader))
            for i, data in enumerate(val_loader, 0):
                x = np.array([data[i][0] for i in range(len(data))])
                y = np.array([data[i][1] for i in range(len(data))])
                batch_acc = self.accuracy(x, y)
                epoch_val_acc[i] = batch_acc
                                    
            val_acc.append(epoch_val_acc.mean())
            print('\r'+"Epoch %d - train accuracy : %.4f - val accuracy: %.4f"%(epoch, train_acc[-1], val_acc[-1]))
        
        return train_acc, val_acc
    

    def accuracy(self, X, y, f=None):
        """
        Computes the accuracy of the networks' predictions
        
        Parameters
        ----------
        X: numpy array
            Test set of shape (samples, N, D)
        y: numpy array
            Prediction targets of shape (samples,)
            This must not be provided as one hot encoded
        f: Tensor, optional
            If provided should be Network.forward(X)
            
        Returns
        -------
        accuracy: float
            Fraction of correctly labelled samples over the total
            number of samples
        """
        
        if f is None:
            f = self.forward(X)
        y_pred = np.argmax(f.elem, axis=0)
        errors = (y!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)
        return accuracy
  

    ### ORIGINAL (same batch for right and òeft sweep) ###
    def sweep(self, X, y, f, lr):
        """
        Makes an optimization "sweep" , consisting of optimizing each pair
        of adjacent Tensors As[i] As[i+1] from i=0 to i=N-1
        
        Parameters
        ----------
        X : numpy array of float
            Shape (batch_size, N, D) (see init parameters for better explanation)
        y: numpy array of int
            Prediction targets of shape (batch_size,)
        f: Tensor
            Result of net.forward(X)
        lr: float in (0,1]
            Learning rate that multiplies the gradient
            
        Returns
        -------
        f: Tensor
            Equivalent to self.forward(X) after the optimization sweep
        """
        
        batch_size = len(y)

        # compute one hot encoding of the target
        one_hot_y = np.zeros((y.size, self.L))
        one_hot_y[np.arange(y.size),y] = 1
        y = one_hot_y.T
        
        
        self.l_cum_contraction = [] # init left cumulative contraction array
        # sweep from left to right
        for i in range(self.N-1):
            #print("\nright sweep step ",i)
            f = self.r_sweep_step(f, y, lr, batch_size)
        
        self.r_cum_contraction = [] # init right cumulative contraction array
        # sweep from right to left
        for i in range(self.N-1):
            #print("\nleft sweep step ",self.N-1-i)
            f = self.l_sweep_step(f, y, lr, batch_size)
        
        return f


    def r_sweep(self, X, y, f, lr):
        """
        Makes an optimization "sweep" , consisting of optimizing each pair
        of adjacent Tensors As[i] As[i+1] from i=0 to i=N-1
        
        Parameters
        ----------
        X : numpy array of float
            Shape (batch_size, N, D) (see init parameters for better explanation)
        y: numpy array of int
            Prediction targets of shape (batch_size,)
        f: Tensor
            Result of net.forward(X)
        lr: float in (0,1]
            Learning rate that multiplies the gradient
            
        Returns
        -------
        f: Tensor
            Equivalent to self.forward(X) after the optimization sweep
        """
        
        batch_size = len(y)

        # compute one hot encoding of the target
        one_hot_y = np.zeros((y.size, self.L))
        one_hot_y[np.arange(y.size),y] = 1
        y = one_hot_y.T
        
        # init left cumulative contraction array
        self.l_cum_contraction = []

        # sweep from left to right
        for i in range(self.N-1):
            #print("\nright sweep step ",i)
            f = self.r_sweep_step(f, y, lr, batch_size)
        return f
    

    def l_sweep(self, X, y, f, lr):
        """
        Makes an optimization "sweep", consisting of optimizing each pair
        of adjacent Tensors As[i] As[i-1] from i=N-1 to i=0
        
        Parameters
        ----------
        X : numpy array of float
            Shape (batch_size, N, D) (see init parameters for better explanation)
        y: numpy array of int
            Prediction targets of shape (batch_size,)
        f: Tensor
            Result of net.forward(X)
        lr: float in (0,1]
            Learning rate that multiplies the gradient
            
        Returns
        -------
        f: Tensor
            Equivalent to self.forward(X) after the optimization sweep
        """
        
        batch_size = len(y)

        # compute one hot encoding of the target
        one_hot_y = np.zeros((y.size, self.L))
        one_hot_y[np.arange(y.size),y] = 1
        y = one_hot_y.T
        
        # init right cumulative contraction array
        self.r_cum_contraction = []

        # sweep from right to left
        for i in range(self.N-1):
            #print("\nleft sweep step ",self.N-1-i)
            f = self.l_sweep_step(f, y, lr, batch_size)
        return f
            

    def r_sweep_step(self, f, y, lr, batch_size):
        """
        Makes a step of the optimization "sweep", consisting in the optimization of
        a pair of Tensors As[i] As[i+1]
        
        Parameters
        ----------
        f: Tensor
            Equivalent to self.forward(X)
        y: numpy array of int
            One hot encoded version of the prediction targets
            Shape is (batch_size,L)
        lr: float in [0,1]
            Learning rate
        batch_size: int
            Number of samples in a batch
            
        Returns
        -------
        f: Tensor
            Equivalent to self.forward(X) after optimization step
        """

        # ID of the node A at which the output of the net is computed
        l = self.l_pos
        
        B = contract(self.As[l], self.As[l+1], "right", "left")    
        
        # computing all elements for delta_B
        # Contributions:
        # - TX[l]    (always))
        # - TX[l+1]    (always))
        # - left_contribution    (for l > 0)
        # - r_cum_contraction[-(l+2)]    (for l < N-2)
        # - y-f    (always)
        
        phi = contract(self.TX[l], self.TX[l+1], common="b")
        
        if l==0:
            # tensor product with broadcasting on batch axis
            phi = contract(phi, self.r_cum_contraction[l+2], common = "b")
        
        elif (l > 0) and (l<(self.N-2)):
            # compute new term for the left contribute
            new_contribution = contract(self.As[l-1], self.TX[l-1], contracted='d'+str(l-1))
            if l==1:
                # define l_cum_contraction (['right','b'])
                self.l_cum_contraction.append(new_contribution)
            else:
                # update l_cum_contraction (['right','b'])
                tmp = contract(self.l_cum_contraction[-1], new_contribution, 'right', 'left', common='b')
                self.l_cum_contraction.append(tmp) 
            
            ### ORIGINAL ###
            #circle_contraction = contract(self.r_cum_contraction[l+2], self.l_cum_contraction[-1], 'right', 'left', common='b')
            ## tensor product with broadcasting on batch axis
            #phi = contract(phi, circle_contraction, common = "b")
            phi = contract(phi, self.r_cum_contraction[l+2], common = "b")
            phi = contract(phi, self.l_cum_contraction[-1], common = "b")
            
        else: # case l=N-1           
            # update l_cum_contraction (['right','b'])
            new_contribution = contract(self.As[l-1], self.TX[l-1], contracted='d'+str(l-1))
            tmp = contract(self.l_cum_contraction[-1], new_contribution, 'right', 'left', common='b')
            self.l_cum_contraction.append(tmp) 
            
            # tensor product with broadcasting on batch axis
            phi = contract(phi, self.l_cum_contraction[-1], common = "b")
            
        ######################################################
        y_pred = np.argmax(f.elem, axis=0) 
        y_target = np.argmax(y, axis=0) 
        #print("Target: ", y_target)
        #print("Prediction (before optim.): ", y_pred)
        
        errors = (y_target!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)
        MSE = ((y-f.elem)**2).mean()
        #print("Accuracy (before optim.): ", accuracy)
        #print("MSE (before optim.): ", MSE)
        ######################################################
        
        #print('f_in: ', np.abs(f.elem).sum()) # debug
        f.elem = y-f.elem # overwrite f with (target - prediction)
  
        deltaB = contract(f, phi, contracted="b")
        # gradient clipping -> rescale all elements of the gradient so that the
        # norm does not exceed the sum of the absolute values of B's entries
        B_measure = np.abs(B.elem).sum()

        if np.abs(deltaB.elem).sum() > B_measure:
            deltaB.elem /= np.abs(deltaB.elem).sum()/B_measure
        deltaB.elem *= lr # multiply gradient for learning rate
        #print('DeltaB: ', np.abs(deltaB.elem).sum()) # debug

        ### ORIGINAL ###
        ## change left and right indices 
        #left_index = deltaB.ax_to_index('left')
        #right_index = deltaB.ax_to_index('right')
        #deltaB.axes_names[left_index] = 'right'
        #deltaB.axes_names[right_index] = 'left'
        if l==0:
            left_index = deltaB.ax_to_index('left')
            deltaB.axes_names[left_index] = 'right'
        elif (l > 0) and (l<(self.N-2)):
            left_index = deltaB.ax_to_index('left')
            right_index = deltaB.ax_to_index('right')
            deltaB.axes_names[left_index] = 'right'
            deltaB.axes_names[right_index] = 'left'
        else: 
            right_index = deltaB.ax_to_index('right')
            deltaB.axes_names[right_index] = 'left'


        #print('B: \t', np.abs(B.elem).sum())
        # just trying to regularize
        B = B + deltaB # update B
        #print('B (after update): ', np.abs(B.elem).sum())
        
        # compute new output of the net (out is like f, but with new A weights)
        out = contract(B, self.TX[l], contracted='d'+str(l))
        out = contract(out, self.TX[l+1], contracted='d'+str(l+1), common='b')
        if l == 0:
            # no left term
            out = contract(out, self.r_cum_contraction[l+2], 'right', 'left', common = "b")
        elif (l > 0) and (l<(self.N-2)):
            # both left and right term
            out = contract(self.l_cum_contraction[-1], out, 'right', 'left', common = "b")
            out = contract(out, self.r_cum_contraction[l+2], 'right', 'left', common = "b")
        else:
            # no right term
            out = contract(self.l_cum_contraction[-1], out, 'right', 'left', common = "b")
        
        ### ORIGINAL (following line was present) ###
        #out = partial_trace(out, 'right', 'left') # close the circle
        
        ######################################################
        y_pred = np.argmax(out.elem, axis=0) 
        #print("Prediction (after optim.): ", y_pred)
        errors = (y_target!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)
        MSE = ((y-out.elem)**2).mean()
        #print("Accuracy (after optim.): ", accuracy)
        #print("MSE (after optim.): ", MSE)
        ######################################################
        
        ### ORIGINAL ###
        ## reconstruct optimized network tensors
        #B.aggregate(axes_names=['d'+str(l),'left'], new_ax_name='i')
        #B.aggregate(axes_names=['d'+str(l+1),'right','l'], new_ax_name='j')
        #B.transpose(['i','j'])
        if l == 0:
            # no left index
            B.aggregate(axes_names=['d'+str(l)], new_ax_name='i')
            B.aggregate(axes_names=['d'+str(l+1),'right','l'], new_ax_name='j')
        elif (l > 0) and (l<(self.N-2)):
            # both left and right indexes
            B.aggregate(axes_names=['d'+str(l),'left'], new_ax_name='i')
            B.aggregate(axes_names=['d'+str(l+1),'right','l'], new_ax_name='j')
        else:
            # no right index
            B.aggregate(axes_names=['d'+str(l),'left'], new_ax_name='i')
            B.aggregate(axes_names=['d'+str(l+1),'l'], new_ax_name='j')        
        B.transpose(['i','j'])


        # use SVD to decompose B in As[l] and As[l+1]
        # l dimension now is on As[l+1]
        ### ORIGINAL ###
        #self.As[l], self.As[l+1] = tensor_svd(B)
        self.As[l], self.As[l+1] = self.tensor_svd(B)

        # update position of l to the right
        self.l_pos += 1
        
        return out
    

    def l_sweep_step(self, f, y, lr, batch_size):
        """
        Makes a step of the optimization "sweep", consisting in the optimization of
        a pair of Tensors As[i] As[i+1]
        
        Parameters
        ----------
        f: Tensor
            Equivalent to self.forward(X)
        y: numpy array of int
            One hot encoded version of the prediction targets
            Shape is (batch_size,L)
        lr: float in [0,1]
            Learning rate
        batch_size: int
            Number of samples in a batch
            
        Returns
        -------
        f: Tensor
            Equivalent to self.forward(X) after optimization step
        """
        
        # ID of the node A at which the output of the net is computed
        l = self.l_pos

        B = contract(self.As[l-1], self.As[l], "right", "left")
        
        # (always true)
        # computing all elements for delta_B
        # Contributions:
        # - TX[l]    (always))
        # - TX[l+1]    (always))
        # - left_contribution    (for l > 0)
        # - r_cum_contraction[-(l+2)]    (for l < N-2)
        # - y-f    (always)
        
        phi = contract(self.TX[l-1], self.TX[l], common="b")
        
        if l == self.N-1:
            # tensor product with broadcasting on batch axis
            phi = contract(phi, self.l_cum_contraction[-1], common = "b")
            
        elif (l > 1) and (l<(self.N-1)):
            # compute new term for the right contribute
            new_contribution = contract(self.As[l+1], self.TX[l+1], contracted='d'+str(l+1))
            
            if l==self.N-2:
                # define r_cum_contraction (['left','b'])
                self.r_cum_contraction.append(new_contribution)
            else:
                # update r_cum_contraction (['left','b'])
                tmp = contract(new_contribution, self.r_cum_contraction[-1], 'right', 'left', common='b')
                self.r_cum_contraction.append(tmp) 
            
            ### ORIGINAL ###
            #circle_contraction = contract(self.r_cum_contraction[-1], self.l_cum_contraction[l-2], 'right', 'left', common='b')
            ## tensor product with broadcasting on batch axis
            #phi = contract(phi, circle_contraction, common = "b")
            phi = contract(phi, self.r_cum_contraction[-1], common = "b")
            phi = contract(phi, self.l_cum_contraction[l-2], common = "b")

        elif l==1:
            new_contribution = contract(self.As[l+1], self.TX[l+1], contracted='d'+str(l+1))
            tmp = contract(new_contribution, self.r_cum_contraction[-1], 'right', 'left', common='b')
            self.r_cum_contraction.append(tmp)
            phi = contract(phi, self.r_cum_contraction[-1], common = "b")
        else:
            print('l: ', l)
            print("This should not happen")           
            
        ######################################################
        y_pred = np.argmax(f.elem, axis=0) 
        y_target = np.argmax(y, axis=0) 
        #print("Target: ", y_target)
        #print("Prediction (before optim.): ", y_pred)
        
        errors = (y_target!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)
        MSE = ((y-f.elem)**2).mean()
        #print("Accuracy (before optim.): ", accuracy)
        #print("MSE (before optim.): ", MSE)
        ######################################################
        
        #print('f: ', np.abs(f.elem).sum())
        f.elem = y-f.elem  # overwrite f with (target - prediction)

        deltaB = contract(f, phi, contracted="b")

        # gradient clipping -> rescale all elements of the gradient so that the
        # norm does not exceed the sum of the absolute values of B's entries
        B_measure = np.abs(B.elem).sum()
        if np.abs(deltaB.elem).sum() > B_measure:
            deltaB.elem /= np.abs(deltaB.elem).sum()/B_measure
        deltaB.elem *= lr

        ### ORIGINAL ###
        ## change left and right indices 
        #left_index = deltaB.ax_to_index('left')
        #right_index = deltaB.ax_to_index('right')
        #deltaB.axes_names[left_index] = 'right'
        #deltaB.axes_names[right_index] = 'left'
        if (l == 1):
            left_index = deltaB.ax_to_index('left')
            deltaB.axes_names[left_index] = 'right'
        elif (l > 1) and (l < (self.N-1)):
            left_index = deltaB.ax_to_index('left')
            right_index = deltaB.ax_to_index('right')
            deltaB.axes_names[left_index] = 'right'
            deltaB.axes_names[right_index] = 'left'
        else: 
            right_index = deltaB.ax_to_index('right')
            deltaB.axes_names[right_index] = 'left'        
        

        #print('B: \t', np.abs(B.elem).sum()), 
        #print('deltaB: ', np.abs(deltaB.elem).sum())

        # update B     
        B = B + deltaB
        
        #print('B.elem.sum() (after update): ', B.elem.sum())
            
        # compute new output of the net (out is like f, but with new A weights)
        out = contract(B, self.TX[l-1], contracted='d'+str(l-1))
        out = contract(out, self.TX[l], contracted='d'+str(l), common='b')
        
        if l == self.N-1:
            # no right term
            out = contract(self.l_cum_contraction[-1], out, 'right', 'left', common = "b") # ok
        
        elif (l > 1) and (l<(self.N-1)):
            # both right and left terms
            out = contract(self.l_cum_contraction[l-2], out, 'right', 'left', common = "b") # ok
            out = contract(out, self.r_cum_contraction[-1], 'right', 'left', common = "b") # ok
            
        else: # l=1 case
            # only right case
            out = contract(out, self.r_cum_contraction[-1], 'right', 'left', common = "b") 
        
        ### ORIGINAL ###
        #out = partial_trace(out, 'right', 'left') # close the circle
        #print("f (old B): ", np.abs(out.elem).sum())
           
        ######################################################
        y_pred = np.argmax(out.elem, axis=0) 
        #print("Prediction (after optim.): ", y_pred)
        errors = (y_target!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)
        MSE = ((y-out.elem)**2).mean()
        #print("Accuracy (after optim.): ", accuracy)
        #print("MSE (after optim.): ", MSE)
        ######################################################

        ### ORIGINAL ###
        ## reconstruct optimized network tensors
        #B.aggregate(axes_names=['d'+str(l-1),'left','l'], new_ax_name='i')
        #B.aggregate(axes_names=['d'+str(l),'right'], new_ax_name='j')
        #B.transpose(['i','j'])
        if l == (self.N-1):
            # no right index
            B.aggregate(axes_names=['d'+str(l-1),'left','l'], new_ax_name='i')
            B.aggregate(axes_names=['d'+str(l)], new_ax_name='j') 
        elif (l > 1) and (l<(self.N-1)):
            # both left and right indexes
            B.aggregate(axes_names=['d'+str(l-1),'left','l'], new_ax_name='i')
            B.aggregate(axes_names=['d'+str(l),'right'], new_ax_name='j')
        else:  
            # no left index
            B.aggregate(axes_names=['d'+str(l-1),'l'], new_ax_name='i')
            B.aggregate(axes_names=['d'+str(l),'right'], new_ax_name='j')     
        B.transpose(['i','j'])


        # use SVD to decompose B in As[l-1] and As[l]
        # l dimension now is on As[l-1]
        ### ORIGINAL ###
        #self.As[l-1], self.As[l] = tensor_svd(B, inverse=True)
        self.As[l-1], self.As[l] = self.tensor_svd(B, inverse=True)
       
        # update position of l to the left
        self.l_pos -= 1

        return out
          

    def tensor_svd(self, T, threshold=0.999, inverse=False):
        """
        Performs Singular Value Decomposition (SVD) on a 2D Tensor
        
        Parameters
        ----------
        T : Tensor
        threshold: float in (0,1)
            Minimum of variance percentage to be explained after truncation
            
        Returns
        -------
        TU : Tensor
            Tensor correspondig to matrix U of U S Vh decomposition
            If original shape was (a,b), it has shape (a,m')
        TSVh : Tensor
            Tensor correspondig to matrix S Vh of U S Vh decomposition
            If original shape was (a,b), it has shape (m',b)
            
        Raises
        ------
        TypeError
            If T is not a Tensor
        ValueError
            If T is not a 2D Tensor
            
        Notes
        -----
        Currently the minimum bond dimension is set to 10. Adaptive thresholding 
        technique is still to be tested in depth before removing this constraint.
        """
        
        if type(T) != Tensor:
            raise TypeError("This function only support object from the class Tensor")

        if len(T.shape) != 2:
            raise ValueError("This function only support a 2D tensors")

        debug = False
        if debug:
            print('\nSVD debug: ', debug)
            print('T.elem.shape: ', T.elem.shape)
            print('T.elem.sum(): ', T.elem.sum())
        # perform SVD of T elements
        U, S, Vh = np.linalg.svd(copy.deepcopy(T.elem))

        #print("S.sum(): ", S.sum())
        # compute adaptive bond dimension
        cumulative_variance_explained = np.cumsum(S)/S.sum()
        #print(cumulative_variance_explained)
        index = np.argmax(cumulative_variance_explained>threshold)


        ### ORIGINAL (check "NEW")###
        """
        m = T.aggregations['i']['left']

        if debug:
            print('m: ', m)
        #####################################
        #m_new = max(10,min(index,m))
        m_new = m # debug
        #print('SVD precision: ', cumulative_variance_explained[m-1])
        #####################################
        # truncate tensors according to the new bond dimension
        Vh = Vh[:m_new,:]
        U = U[:,:m_new]
        S = np.eye(m_new, m_new)*S[:m_new]
        Sqrt = np.sqrt(S)
        """

        if inverse is False:

            # NEW: truncate only matrices with left or right indexes
            # (if first or last case -> S.shape = (2,) )
            if (self.l_pos == 0):
                m = len(S)
                Vh = Vh[:m,:]
                S = np.eye(m, m) * S
            elif (self.l_pos > 0) and (self.l_pos < self.N-2):
                m = T.aggregations['i']['left']
                U = U[:,:m]
                Vh = Vh[:m,:]
                S = np.eye(m, m) * S[:m]
            else:
                m = len(S)
                U = U[:,:m]
                S = np.eye(m, m) * S

            Sqrt = np.sqrt(S)


            #SVh = np.dot(S, Vh) # new is np.newaxis
            SVh = np.dot(Sqrt, Vh)
            U = np.dot(U,Sqrt)
                
            # building new tensors
            TU = Tensor(elem=U, axes_names=['i','right'])
            TSVh = Tensor(elem=SVh, axes_names=['left','j'])
            TU.aggregations['i'] = T.aggregations['i']
            TSVh.aggregations['j'] = T.aggregations['j']

            # retrieving original dimensions
            TU.disaggregate('i')
            TSVh.disaggregate('j')

            return TU, TSVh
        
        else:

            # NEW: truncate only matrices with left or right indexes
            if (self.l_pos == self.N-1):
                m = len(S)
                U = U[:,:m]
                S = np.eye(m, m) * S
            elif (self.l_pos > 1) and (self.l_pos < self.N-1):
                m = T.aggregations['i']['left']
                U = U[:,:m]
                Vh = Vh[:m,:]
                S = np.eye(m, m) * S[:m]
            else:
                m = len(S)
                Vh = Vh[:m,:]
                S = np.eye(m, m) * S

            Sqrt = np.sqrt(S)

            #US = np.dot(U,S)
            US = np.dot(U,Sqrt)
            SVh = np.dot(Sqrt, Vh)

            # building new tensors
            TUS = Tensor(elem=US, axes_names=['i','right'])
            TSVh = Tensor(elem=SVh, axes_names=['left','j'])
            TUS.aggregations['i'] = T.aggregations['i']
            TSVh.aggregations['j'] = T.aggregations['j']

            # retrieving original dimensions
            TUS.disaggregate('i')
            TSVh.disaggregate('j')

            return TUS, TSVh#, cumulative_variance_explained[m-1]

