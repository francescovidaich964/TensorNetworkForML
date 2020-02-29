
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
    act_fn: str
        Name of the activation function on the output (can be 'linear', 
        'cross_ent', 'full_cross_ent') 
    loss_fn: str
        Name of the loss function to use during training (can be 'MSE', 
        'cross_entropy', 'full_cross_ent')
    T: float
        Value of the temperature used to compute the softmax
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
        Computes predictions for input X (without using the activation function)
    train(train_loader, val_loader, lr, lr, n_epochs=10, weight_dec=0.001, L2_flag=True, debug=False)
        Trains the network 
    accuracy(x, y, f=None)
        Computes the accuracy of the networks' predictions
    sweep(X, y, f, lr, weight_dec, left_dir=False, L2_flag=True, var_hist=None, debug=False)
        Makes an optimization "sweep" , consisting of optimizing each pair of 
        adjacent Tensors As[i] As[i+1] from i=0 to i=N-1 (or in the other direction)
    sweep_step(f, y, lr, batch_size, weight_dec, left_dir=False, L2_flag=True, var_hist=None, debug=False)
        Makes a step of the optimization "sweep", consisting in the optimization of
        a pair of Tensors As[i] As[i+1] (or As[i-1] A[i] if left_dir=True)
    update_B(B, f_orig, y, lr, weight_dec, ldf=0, L2_flag=True, var_hist=None, debug=False)
        Compute gradient of the loss function and updates the tensor B
    apply_act_func(f)
        Apply the activation function to the output of the network 
        (can be 'linear', 'cross_ent', 'full_cross_ent') 
    compute_loss_derivate(f, y):
        Compute the derivate of the loss function with respect to the 
        net output f (can be 'MSE', 'softmax', 'full_cross_ent') 
    tensor_svd(T, left_dir=False, threshold=0.999):
        Performs Singular Value Decomposition (SVD) on a 2D Tensor
    compute_L2_reg(B, weight_dec=0.001, left_dir=False):
        Compute the L2 regularization term and its derivate with respect to B
        
    Notes
    -----
    This class implements a linear MPS Tensor Network, meaning that there are two tensors
    at the extremes of the net connected to only 1 othe thensor instead of 2, so they have 
    a different shape and are handled a bit differently.
    Moreover, the training function changes batch every time the sweep changes direction.
    """
    
    def __init__(self, N, M, D=2, L=10, T=0.1, normalize=False, calibration_X=None, act_fn='linear', loss_fn='cross_entropy', check=False):
        """
        Parameters
        ----------
        N: int
            First dimension of the input (e.g. number of pixels for an image)
        M: int
            Initial bond dimension
        D: int, optional
            Second dimension of the input (e.g. number of color channels or
            embedding dimension)
        L: int, optional
            Number of possible labels
        T: float, optional
            Value of the temperature used to compute the softmax
        normalize: bool, optional
            If True, normalizes the weights so that the expected output is of order 1
            for inputs X with all entries in [0,1]
        calibration_X: numpy array, optional
            Sample of shape (batch_size, N, D) used to calibrate the weights 
            of the network before training
        act_fn: str, optional
            Name of the activation function on the output (can be 'linear', 
            'cross_ent', 'full_cross_ent') 
        loss_fn: str, optional
            Name of the loss function to use during training (can be 'MSE', 
            'cross_entropy', 'full_cross_ent')
        check: bool, optional
            Print results of the calibration to check that it is working properly
        """
        
        self.N = N
        self.D = D
        self.L = L
        self.M = M
        self.T = T  # temperature for softmax
        
        self.As = []

        # Initial position of the tensor with additional dimension for the output of the net
        self.l_pos = 0
        
        # Choose output activation function
        possible_act_fn = ['linear', 'sigmoid', 'softmax']
        assert act_fn in possible_act_fn, "Please select an activation function between 'linear', 'sigmoid', 'softmax'"
        self.act_fn = act_fn

        # Choose output activation function
        possible_loss_fn = ['MSE', 'cross_entropy', 'full_cross_ent']
        assert loss_fn in possible_loss_fn, "Please select a loss function between 'MSE', 'cross_entropy', 'full_cross_ent'"      
        self.loss_fn = loss_fn


        if normalize:
            print('Normalizing weights...')
            # output goes like [M E(A) E(x) D]^N
            # E(A) expected value of entry A (tensor) distributed uniformly in [0,1] -> 0.5
            # E(x) expected value of x = cos(pi/2 * u) (or sin(pi/2 * u)), u in [0,1] -> 0.64
            scale = float(self.M)*0.5*0.64*self.D
            print('Scaling factor: %.2f'%scale)
            
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
            
            print('\nCalibrating weights on dataset...')

            # compute the order of magnitude of the output
            f = self.forward(X)
            f_max = np.abs(f.elem).max().astype('float')
            F2 = f_max**(1./self.N) # factor for rescaling
            if check:
                print('f_max for random input of %d samples : '%(B),f_max)

            print("Rescaling factor for calibration: ", F2)
            for i in range(self.N):
                self.As[i].elem = self.As[i].elem/F2
                
            # compute the new order of magnitude of the output (should be 1)
            f = self.forward(X)  
            f_max = np.abs(f.elem).max().astype('float')
            if check:
                print('f_max for random input of %d samples (after): '%(B),f_max)
            
        else:
            # without normalization each entry of each A is a random number in [0,1]
            self.As.append(Tensor(shape=[L,M,D], axes_names=['l','right','d0']))
            for i in range(1,N-1):
                self.As.append(Tensor(shape=[M,M,D], axes_names=['left','right','d'+str(i)]))
            self.As.append(Tensor(shape=[M,D], axes_names=['left','d'+str(N-1)]))

        return
    


    def forward(self, X):
        """
        Computes predictions for input X (without using the activation function)
        Function behaves differently depending on the position of the l index
        
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
        self.TX = TX              
        
        A_TX = [contract(self.As[i], TX[i], contracted='d'+str(i)) for i in range(self.N)]
        cum_contraction = []
        
        # Compute forward with L or R cum_contractions depending on the sweep to do
        if self.l_pos == 0:

            # Prepare for right sweep
            cum_contraction.append(A_TX[-1])
            for j in range(1,self.N): 
                tmp_cum = copy.deepcopy(cum_contraction[-1])
                tmp_cum = contract(A_TX[-(j+1)], tmp_cum, 'right', 'left', common='b')
                cum_contraction.append(tmp_cum)

            self.r_cum_contraction = cum_contraction[::-1]
            self.l_cum_contraction = None
            return self.r_cum_contraction[0]

        elif self.l_pos == self.N-1:

            # Prepare for left sweep
            cum_contraction.append(A_TX[0])
            for j in range(1,self.N): 
                tmp_cum = copy.deepcopy(cum_contraction[-1])
                tmp_cum = contract(tmp_cum, A_TX[j], 'right', 'left', common='b')
                cum_contraction.append(tmp_cum)

            self.r_cum_contraction = None
            self.l_cum_contraction = cum_contraction #left are already ordered
            return self.l_cum_contraction[-1]

        else:
            raise Exception('### Error ###\n l =',l,' -> forward should not be called if l has an intermediate position')


    def train(self, train_loader, val_loader, lr, n_epochs=10, weight_dec=0.001, L2_flag=True, debug=False):
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
        weight_dec: float, optional
            value of weight decay for regularization (set 0 to disable this feature)
        L2_flag: bool, optional
            Set type of regularization (True -> L2 regularization added to loss function)
                                       (False -> decrease B elements by a factor equal to weight_dec)
        debug: bool, optional
            If True, the function returns the evolution of many variables 
            If False, the function returns the evolution of accuracy and MAE
            
        Returns
        -------
        val_acc: list of float
            List of accuracies and validation set for each epoch
        var_hist: numpy array
            Evolution of many variables during training

        Notes
        -----
        At the moment if debug == False, var_hist will contain the evolution of:
                np.abs(B.elem).mean()
                np.abs(deltaB.elem).mean()
                accuracy
                np.abs(f_orig.elem).mean()
                MAE
                L2_loss_term
                np.abs(L2_gradient.elem).mean()
        It is possible to change these observed variables in the method self.update_B()

        """
        
        val_acc = []
        var_hist = []

        print("\n --- TRAINING PROCEDURE ---")

        for epoch in range(n_epochs):
        
            epoch_train_acc = np.zeros(len(train_loader))
            
            ######  DEBUG  ######
            if debug:
                var_hist.append([[],[],[],[],[],[],[]])
            else:
                var_hist.append([[],[]])
            #####################


            # Training
            for i, data in enumerate(train_loader, 0):
                x = np.array([data[i][0] for i in range(len(data))])
                y = np.array([data[i][1] for i in range(len(data))])
                
                f = self.forward(x)
                batch_acc = self.accuracy(x, y, f) # compute accuracy before batch optimization
                epoch_train_acc[i] = batch_acc

                # Perform a sweep in the correct direction
                left_dir = (self.l_pos == self.N-1)  # True for left direction
                f = self.sweep(x, y, f, lr, weight_dec, L2_flag=L2_flag, left_dir=left_dir, var_hist=var_hist[epoch], debug=debug)
                
                print('\r'+"Epoch %d/%d - train accuracy : %.4f - completed : %.2f "%(epoch, n_epochs, epoch_train_acc[i], (i+1)*100/len(train_loader))+'%', end=' ')

            
            # Validation
            epoch_val_acc = np.zeros(len(val_loader))
            for i, data in enumerate(val_loader, 0):
                x = np.array([data[i][0] for i in range(len(data))])
                y = np.array([data[i][1] for i in range(len(data))])
                batch_acc = self.accuracy(x, y)
                epoch_val_acc[i] = batch_acc
                                    
            val_acc.append(epoch_val_acc.mean())
            print('\r'+"Epoch %d/%d - train accuracy : %.4f - val accuracy: %.4f"%(epoch, n_epochs, epoch_train_acc.mean(), val_acc[-1]))
        

        return val_acc, np.array(var_hist)
    


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
  


    def sweep(self, X, y, f, lr, weight_dec, L2_flag=True, left_dir=False, var_hist=None, debug=False):
        """
        Makes an optimization "sweep" , consisting of optimizing each pair
        of adjacent Tensors As[i] As[i+1] from i=0 to i=N-1 (or in the other direction)
        
        Parameters
        ----------
        X : numpy array of float
            Shape (batch_size, N, D) (see init parameters for better explanation)
        y: numpy array of int
            Prediction targets of shape (batch_size,)
        f: Tensor
            Result of net.forward(X) (i.e. the non activated output)
        lr: float in (0,1]
            Learning rate that multiplies the gradient
        weight_dec: float
            value of weight decay for regularization (set 0 to disable this feature)
        L2_flag: bool, optional
            Set type of regularization (True -> L2 regularization added to loss function)
                                       (False -> decrease B elements by a factor equal to weight_dec)
        left_dir: bool optional
            Flag that indicates the direction of the sweep
        var_hist: list, optional
            Empty list that will contain the evolution of the observed variables
        debug: bool, optional
            If True, var_hist contains the evolution of many variables 
            If False, var_hist contains the evolution of accuracy and MAE    
            
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
        
        # Reinitialize right or left cum_contractions (depending on sweep direction)
        if left_dir:
            self.r_cum_contraction = []
        else:
            self.l_cum_contraction = []

        # Perform a sweep step in the given direction (ex: if inverse -> left dir)
        for i in range(self.N-1):
            f = self.sweep_step(f, y, lr, batch_size, weight_dec, L2_flag=L2_flag, 
                                left_dir=left_dir, var_hist=var_hist, debug=debug)
        
        return f



    def sweep_step(self, f, y, lr, batch_size, weight_dec, L2_flag=True, left_dir=False, var_hist=None, debug=False):
        """
        Makes a step of the optimization "sweep", consisting in the optimization of
        a pair of Tensors As[i] As[i+1] (or As[i-1] A[i] if left_dir=True)
        
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
        weight_dec: float
            value of weight decay for regularization (set 0 to disable this feature)
        L2_flag: bool, optional
            Set type of regularization (True -> L2 regularization added to loss function)
                                       (False -> decrease B elements by a factor equal to weight_dec)
        left_dir: bool optional
            Flag that indicates the direction of the sweep
        var_hist: list, optional
            list that contains the evolution of the observed variables
        debug: bool, optional
            If True, var_hist contains the evolution of many variables 
            If False, var_hist contains the evolution of accuracy and MAE    

        Returns
        -------
        out: Tensor
            Equivalent to self.forward(X) after optimization step
        """


        # ID of the node A at which the output of the net is computed
        l = self.l_pos

        # To execute the update with the correct indexes, we will use
        # a binary int value that will be added to them if inverse = True
        ldf = int(left_dir)   # "Left Direction Flag"
        
        # Compute the B tensor at current l      
        B = contract(self.As[l-ldf], self.As[l+1-ldf], "right", "left")    

        # Perform the update of B
        B = self.update_B(B, f, y, lr, weight_dec, L2_flag=L2_flag, ldf=ldf,
                          var_hist=var_hist, debug=debug)



        ##### Compute new output of the network using the updated B #####

        out = contract(B, self.TX[l-ldf], contracted='d'+str(l-ldf))
        out = contract(out, self.TX[l+1-ldf], contracted='d'+str(l+1-ldf), common='b')
        
        # R and L cum_contractions are used differently for R and L sweeps
        #  -> handle the two cases in different ways
        if left_dir == False:  # Sweep to the R direction

            if l == 0: 
                # No left term
                out = contract(out, self.r_cum_contraction[l+2], 'right', 'left', common = "b")
            elif (l > 0) and (l < (self.N-2)):
                # Both left and right term
                out = contract(self.l_cum_contraction[-1], out, 'right', 'left', common = "b")
                out = contract(out, self.r_cum_contraction[l+2], 'right', 'left', common = "b")
            else:
                # No right term
                out = contract(self.l_cum_contraction[-1], out, 'right', 'left', common = "b")
        
        else:  # Sweep to the L direction

            if l == self.N-1:
                # No right term
                out = contract(self.l_cum_contraction[-3], out, 'right', 'left', common = "b")
            elif (l > 1) and (l<(self.N-1)):
                # Both right and left terms
                out = contract(self.l_cum_contraction[l-2], out, 'right', 'left', common = "b")
                out = contract(out, self.r_cum_contraction[-1], 'right', 'left', common = "b")
            else:
                # No right term
                out = contract(out, self.r_cum_contraction[-1], 'right', 'left', common = "b") 



        # Aggregate B indexes to build a 2D tensor for SVD
        if left_dir == False:
         
            if l == 0:
                # No left index
                B.aggregate(axes_names=['d'+str(l)], new_ax_name='i')
                B.aggregate(axes_names=['d'+str(l+1),'right','l'], new_ax_name='j')
            elif (l > 0) and (l < (self.N-2)):
                # Both left and right indexes
                B.aggregate(axes_names=['d'+str(l),'left'], new_ax_name='i')
                B.aggregate(axes_names=['d'+str(l+1),'right','l'], new_ax_name='j')
            else:
                # No right index
                B.aggregate(axes_names=['d'+str(l),'left'], new_ax_name='i')
                B.aggregate(axes_names=['d'+str(l+1),'l'], new_ax_name='j')        

        else:

            if l == 1:
                # No left index
                B.aggregate(axes_names=['d'+str(l-1),'l'], new_ax_name='i')
                B.aggregate(axes_names=['d'+str(l),'right'], new_ax_name='j')
            elif (l > 1) and (l < (self.N-1)):
                # Both left and right indexes
                B.aggregate(axes_names=['d'+str(l-1),'left','l'], new_ax_name='i')
                B.aggregate(axes_names=['d'+str(l),'right'], new_ax_name='j')
            else:
                # No right index
                B.aggregate(axes_names=['d'+str(l-1),'left','l'], new_ax_name='i')
                B.aggregate(axes_names=['d'+str(l)], new_ax_name='j')        


        # Make sure that indexes follows the order ['i','j']
        B.transpose(['i','j'])

        # Use SVD to decompose B in As[l] and As[l+1] (or As[l-1] if left sweep)
        self.As[l-ldf], self.As[l+1-ldf] = self.tensor_svd(B, left_dir)

 

        # Update position of l depending on sweep direction
        if left_dir == False:
            self.l_pos += 1
        else:
            self.l_pos -= 1

        return out
    


    def update_B(self, B, f_orig, y, lr, weight_dec, L2_flag=True, ldf=0, var_hist=None, debug=False):
        """
        Compute gradient of the loss function and updates the tensor B
        
        Parameters
        ----------
        B: Tensor
            Contraction As[i] As[i+1] (or As[i-1] A[i] if ldf=1)
        f_orig: Tensor
            Equivalent to self.forward(X)
        y: numpy array of int
            One hot encoded version of the prediction targets
            Shape is (batch_size,L)
        lr: float in [0,1]
            Learning rate
        weight_dec: float
            value of weight decay for regularization (set 0 to disable this feature)
        L2_flag: bool, optional
            Set type of regularization (True -> L2 regularization added to loss function)
                                       (False -> decrease B elements by a factor equal to weight_dec)
        ldf: int, optional
            Indicates the direction of the sweep (0 for right sweep, 1 for left sweep)
        var_hist: list, optional
            list that contains the evolution of the observed variables
        debug: bool, optional
            If True, var_hist contains the evolution of many variables 
            If False, var_hist contains the evolution of accuracy and MAE    

        Returns
        -------
        B: Tensor
            The updated tensor B
        
        """



        ####### COMPUTING all elements for DELTA_B #######
        #   Contributions (R or L case):
        #   - TX[l] or TX[l-1]   (always)
        #   - TX[l+1] or TX[l]   (always)
        #   - left_contribution or l_cum_contraction[l+2]   (for l > 0)
        #   - r_cum_contraction[-(l+2)] or right_contribution  (for l < N-2)
        #   - y-f    (always)
      

        # Start by contracting the two inputs of B
        l = self.l_pos
        phi = contract(self.TX[l-ldf], self.TX[l+1-ldf], common="b")
        
        # Contract the rest of the network (right and left sweeps are handled differently)
        if ldf == 0:  # Sweep to the R direction

            if l == 0:
                # The entire net is on the right of B -> no need to contract to the left
                phi = contract(phi, self.r_cum_contraction[l+2], common = "b")
            
            elif (l > 0) and (l < (self.N-2)):

                # Compute new term for the left contribute and add to l_cum_contraction
                new_contribution = contract(self.As[l-1], self.TX[l-1], contracted='d'+str(l-1))
                if l==1:
                    self.l_cum_contraction.append(new_contribution)
                else:
                    tmp = contract(self.l_cum_contraction[-1], new_contribution, 'right', 'left', common='b')
                    self.l_cum_contraction.append(tmp) 
                
                # Contract tensors on the right and on the left of B
                phi = contract(phi, self.r_cum_contraction[l+2], common = "b")
                phi = contract(phi, self.l_cum_contraction[-1], common = "b")
                
            elif l == (self.N-2):
                # Compute new term for the left contribute and add to l_cum_contraction
                new_contribution = contract(self.As[l-1], self.TX[l-1], contracted='d'+str(l-1))
                tmp = contract(self.l_cum_contraction[-1], new_contribution, 'right', 'left', common='b')
                self.l_cum_contraction.append(tmp) 
                
                # The entire net is on the left of B -> no need to contract to the right
                phi = contract(phi, self.l_cum_contraction[-1], common = "b")
                
            else:
                raise Exception('### Error ###\n l =',l,' -> position not allowed for right sweep step')

        else:  # Sweep to the L direction

            if l == self.N-1:
                # The entire net is on the right of B -> no need to contract to the left
                phi = contract(phi, self.l_cum_contraction[-3], common = "b")
                
            elif (l > 1) and (l < (self.N-1)):

                # Compute new term for the right contribute and add to r_cum_contraction
                new_contribution = contract(self.As[l+1], self.TX[l+1], contracted='d'+str(l+1))
                if l==self.N-2:
                    self.r_cum_contraction.append(new_contribution)
                else:
                    tmp = contract(new_contribution, self.r_cum_contraction[-1], 'right', 'left', common='b')
                    self.r_cum_contraction.append(tmp) 
                
                # Contract tensors on the right and on the left of B
                phi = contract(phi, self.r_cum_contraction[-1], common = "b")
                phi = contract(phi, self.l_cum_contraction[l-2], common = "b")
             
            elif l == 1:
                # Compute new term for the right contribute and add to r_cum_contraction
                new_contribution = contract(self.As[l+1], self.TX[l+1], contracted='d'+str(l+1))
                tmp = contract(new_contribution, self.r_cum_contraction[-1], 'right', 'left', common='b')
                self.r_cum_contraction.append(tmp)

                # The entire net is on the right of B -> no need to contract to the left
                phi = contract(phi, self.r_cum_contraction[-1], common = "b")

            else:
                raise Exception('### Error ###\n l =',l,' -> position not allowed for left sweep step')


        # Use the activation function for the output of the net 
        f = self.apply_act_func(f_orig)

        # Evaluate current Accuracy and Mean Absolute Error 
        y_pred = np.argmax(f.elem, axis=0) 
        y_target = np.argmax(y, axis=0) 
        errors = (y_target!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)

        MAE = np.abs(y-f.elem).mean()



        # Compute the derivate of the loss function
        loss_der = self.compute_loss_derivate(f, y)

        # Contract output error with the whole (contracted) network without B
        deltaB = contract(loss_der, phi, contracted="b")


        # Swap left and right indixes to retrieve the correct shape of Delta_B
        if l == (0+ldf):
            left_index = deltaB.ax_to_index('left')
            deltaB.axes_names[left_index] = 'right'
        elif (l > (0+ldf)) and (l < (self.N-2+ldf)):
            left_index = deltaB.ax_to_index('left')
            right_index = deltaB.ax_to_index('right')
            deltaB.axes_names[left_index] = 'right'
            deltaB.axes_names[right_index] = 'left'
        else: 
            right_index = deltaB.ax_to_index('right')
            deltaB.axes_names[right_index] = 'left'


        # Perform weight decay (simple or with L2)
        if L2_flag:
            L2_loss_term, L2_gradient = self.compute_L2_reg(B, weight_dec, ldf)
            deltaB -= L2_gradient
        else:
            L2_gradient = copy.deepcopy(B)
            L2_gradient.elem = weight_dec*B.elem
            deltaB -= L2_gradient



        # Store history of variables before the update:
        if var_hist is not None:
            if debug:
                var_hist[0].append(np.abs(B.elem).mean())
                var_hist[1].append(np.abs(deltaB.elem).mean())
                var_hist[2].append(accuracy)
                var_hist[3].append(np.abs(f_orig.elem).mean())
                var_hist[4].append(MAE)
                var_hist[5].append(L2_loss_term)
                var_hist[6].append(np.abs(L2_gradient.elem).mean())
            else:
                var_hist[0].append(accuracy)
                var_hist[1].append(MAE)


        # Gradient clipping -> Rescale all elements of the gradient so that the
        # norm does not exceed the sum of the absolute values of B's entries
        B_measure = np.abs(B.elem).sum()
        if np.abs(deltaB.elem).sum() > B_measure:
            deltaB.elem /= np.abs(deltaB.elem).sum()/B_measure

        # Perform the update of B
        deltaB.elem *= lr
        B = B + deltaB

        return B
    


    def apply_act_func(self, f):
        """
        Apply the activation function (defined in __init__) to the 
        output of the network (can be 'linear', 'cross_ent', 'full_cross_ent') 

        Parameters
        ----------
        f: Tensor
            Equivalent to self.forward(X)


        Returns
        -------
        activation: Tensor
            Output of the network after the Activation function
        """

        # Create a tensor with the same shape and elements of f
        activation = copy.deepcopy(f)

        # Apply activation function
        if self.act_fn == 'linear':  # linear activation is identity transform on f
            pass 
        elif self.act_fn == 'sigmoid':
            activation.elem = 1./(1.+np.exp(-activation.elem/self.T)) # a = 1/(1+e^-f)
        elif self.act_fn == 'softmax':
            label_axis = activation.ax_to_index('l')
            activation.elem = np.exp(activation.elem/self.T)/np.exp(activation.elem/self.T).sum(axis=label_axis)

        return activation



    def compute_loss_derivate(self, f, y):
        """
        Given (activated) outputs and labels, compute the derivate of the loss function
        with respect to the net output f (can be 'MSE', 'softmax', 'full_cross_ent') 

        Parameters
        ----------
        f: Tensor
            Output of the network after the activation function
        y: numpy array of int
            One hot encoded version of the prediction targets
            Shape is (batch_size,L)

        Returns
        -------
        loss_der: Tensor
            derivate of the loss function with respect to the error y-f
        """

        # Create a tensor with the same shape of f
        loss_der = copy.deepcopy(f)

        # Compute loss value
        if self.loss_fn == 'MSE':
            loss_der.elem = y - f.elem
        elif self.loss_fn == 'cross_entropy':
            if self.act_fn == 'softmax':
                print("softmax + cross entropy case")
                loss_der.elem = (y - y*f.elem)/self.T # simplyfied computations
            else:
                loss_der.elem = y/f.elem
        elif self.loss_fn == 'full_cross_ent':
            loss_der.elem[y == 0] = loss_der.elem[y == 0]  - 1
            loss_der.elem = 1./(loss_der.elem+1e-4)

        return loss_der



    def tensor_svd(self, T, left_dir=False, threshold=0.999):
        """
        Performs Singular Value Decomposition (SVD) on a 2D Tensor
        
        Parameters
        ----------
        T : Tensor
            2D Tensor that will be splitted through SVD
        left_dir: bool, optional
            Direction of the sweep
        threshold: float in (0,1), optional
            Minimum of variance percentage to be explained after truncation
            
        Returns
        -------
        TUS : Tensor
            Tensor correspondig to matrix U * sqrt(S)
            If original shape was (a,b), it has shape (a,m')
        TSVh: Tensor
            Tensor correspondig to matrix sqrt(S) * Vh
            If original shape was (a,b), it has shape (m',b)
        
        Raises
        ------
        TypeError
            If T is not a Tensor
        ValueError
            If T is not a 2D Tensor
            
        Notes
        -----
        Adaptive thresholding technique is still to be tested in depth.
        Both U and Vh are multiplied to sqrt(S) for stability purposes.
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

        # Perform SVD of T elements
        U, S, Vh = np.linalg.svd(copy.deepcopy(T.elem))

        # Compute adaptive bond dimension
        cumulative_variance_explained = np.cumsum(S)/S.sum()
        index = np.argmax(cumulative_variance_explained>threshold)


        if left_dir is False:

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

            SVh = np.dot(Sqrt, Vh)
            U = np.dot(U,Sqrt)
                
            # building new tensors
            TU = Tensor(elem=U, axes_names=['i','right'])
            TSVh = Tensor(elem=SVh, axes_names=['left','j'])
            TU.aggregations['i'] = T.aggregations['i']
            TSVh.aggregations['j'] = T.aggregations['j']

            # Retrieving original dimensions 
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

            US = np.dot(U,Sqrt)
            SVh = np.dot(Sqrt, Vh)

            # building new tensors
            TUS = Tensor(elem=US, axes_names=['i','right'])
            TSVh = Tensor(elem=SVh, axes_names=['left','j'])
            TUS.aggregations['i'] = T.aggregations['i']
            TSVh.aggregations['j'] = T.aggregations['j']

            # Retrieving original dimensions 
            TUS.disaggregate('i')
            TSVh.disaggregate('j')

            return TUS, TSVh



    def compute_L2_reg(self, B, weight_dec=0.001, left_dir=False):
        """
        Compute the L2 regularization term and its derivate with respect to B
        
        Parameters
        ----------
        B : Tensor
            2D Tensor that will be splitted through SVD
        weight_dec: float, optional
            value of weight decay for regularization (set 0 to disable this feature)
        left_dir: bool, optional
            Direction of the sweep

        Returns
        -------
        loss_term: float
            Value of the L2 regularization term * weight_dec

        derivate: Tensor
            Derivate of loss_term with respect to B
        

        Notes
        -----
        L2 norm of the weights of the network can be computed by contracting
        all tensors A and then contracting all of it to a copy of itself
        thorugh all inputs (working if self.l_pos is not updated)

        Since the memory can't store the tensor of the contracted network, 
        we contract the tensors over each input before contracting the whole thing

        loss_term scales with N, select a weight_dec to compensatethis behaviour
        """

        # Contract tensors on the right and on the left of B (depending on sweep direction)
        if left_dir == False:  # Case of right sweep (l from 0 to N-2)

            # Contract tensors on the left (if needed)
            if self.l_pos != 0:

                # copy tensor and rename indexes before contracting it to itself
                tmp_tensor = copy.deepcopy(self.As[0])
                tmp_tensor_2 = copy.deepcopy(self.As[0])
                #left_index = tmp_tensor_2.ax_to_index('left')    # As[0] should not have 'left'
                right_index = tmp_tensor_2.ax_to_index('right')
                #tmp_tensor_2.axes_names[left_index] = 'L_2'
                tmp_tensor_2.axes_names[right_index] = 'R_2'

                left_contr = contract(tmp_tensor, tmp_tensor_2, contracted='d0')
                
                # For every other tensor on the left of B, repeat process
                for i in range(1, self.l_pos): 

                    tmp_tensor = copy.deepcopy(self.As[i])
                    tmp_tensor_2 = copy.deepcopy(self.As[i])
                    left_index = tmp_tensor_2.ax_to_index('left')
                    right_index = tmp_tensor_2.ax_to_index('right')
                    tmp_tensor_2.axes_names[left_index] = 'L_2'
                    tmp_tensor_2.axes_names[right_index] = 'R_2'

                    input_contr = contract(tmp_tensor, tmp_tensor_2, contracted='d'+str(i))
                    left_contr = contract(left_contr, input_contr, 
                                          left_contr.ax_to_index(['right','R_2']), 
                                          input_contr.ax_to_index(['left','L_2']))            
            else:
                left_contr = None


            # Contract tensors on the right (if needed)
            if self.l_pos != self.N-2:
                
                # Copy tensor and rename indexes before contracting it to itself
                tmp_tensor = copy.deepcopy(self.As[-1])
                tmp_tensor_2 = copy.deepcopy(self.As[-1])
                left_index = tmp_tensor_2.ax_to_index('left')
                #right_index = tmp_tensor_2.ax_to_index('right')  # As[-1] should not have 'right'
                tmp_tensor_2.axes_names[left_index] = 'L_2'
                #tmp_tensor_2.axes_names[right_index] = 'R_2'
                right_contr = contract(tmp_tensor, tmp_tensor_2, 'd'+str(self.N-1), 'd'+str(self.N-1))

                # For every other tensor on the right of B, repeat process
                for i in range(self.N-2, self.l_pos+1, -1): 

                    tmp_tensor = copy.deepcopy(self.As[i])
                    tmp_tensor_2 = copy.deepcopy(self.As[i])
                    left_index = tmp_tensor_2.ax_to_index('left')
                    right_index = tmp_tensor_2.ax_to_index('right')
                    tmp_tensor_2.axes_names[left_index] = 'L_2'
                    tmp_tensor_2.axes_names[right_index] = 'R_2'

                    # Contract input index of the 2 tensors and then both
                    # its 'left' indexes with the 'right' ones of right_contr
                    input_contr = contract(tmp_tensor, tmp_tensor_2, contracted='d'+str(i))
                    right_contr = contract(input_contr, right_contr, 
                                           input_contr.ax_to_index(['right','R_2']), 
                                           right_contr.ax_to_index(['left','L_2']))                          
            else:
                right_contr = None

        else: 

            # Case of left_sweep (l from N-1 to 1):
            # Contract tensors on the left (if needed)
            if self.l_pos != 1:

                # copy tensor and rename indexes before contracting it to itself
                tmp_tensor = copy.deepcopy(self.As[0])
                tmp_tensor_2 = copy.deepcopy(self.As[0])
                #left_index = tmp_tensor_2.ax_to_index('left')    # As[0] should not have 'left'
                right_index = tmp_tensor_2.ax_to_index('right')
                #tmp_tensor_2.axes_names[left_index] = 'L_2'
                tmp_tensor_2.axes_names[right_index] = 'R_2'

                left_contr = contract(tmp_tensor, tmp_tensor_2, contracted='d0')

                # For every other tensor on the left of B, repeat process
                for i in range(1, self.l_pos-1): 
                    
                    tmp_tensor = copy.deepcopy(self.As[i])
                    tmp_tensor_2 = copy.deepcopy(self.As[i])
                    left_index = tmp_tensor_2.ax_to_index('left')
                    right_index = tmp_tensor_2.ax_to_index('right')
                    tmp_tensor_2.axes_names[left_index] = 'L_2'
                    tmp_tensor_2.axes_names[right_index] = 'R_2'

                    input_contr = contract(tmp_tensor, tmp_tensor_2, contracted='d'+str(i))
                    left_contr = contract(left_contr, input_contr, 
                                          left_contr.ax_to_index(['right','R_2']), 
                                          input_contr.ax_to_index(['left','L_2']))
            else:
                left_contr = None

            # Contract tensors on the left (if needed)
            if self.l_pos != self.N-1:

                # copy tensor and rename indexes before contracting it to itself
                tmp_tensor = copy.deepcopy(self.As[-1])
                tmp_tensor_2 = copy.deepcopy(self.As[-1])
                left_index = tmp_tensor_2.ax_to_index('left')
                #right_index = tmp_tensor_2.ax_to_index('right')  # As[-1] should not have 'right'
                tmp_tensor_2.axes_names[left_index] = 'L_2'
                #tmp_tensor_2.axes_names[right_index] = 'R_2'

                right_contr = contract(tmp_tensor, tmp_tensor_2, contracted='d'+str(self.N-1))

                for i in range(self.N-2, self.l_pos, -1): 

                    tmp_tensor = copy.deepcopy(self.As[i])
                    tmp_tensor_2 = copy.deepcopy(self.As[i])
                    left_index = tmp_tensor_2.ax_to_index('left')
                    right_index = tmp_tensor_2.ax_to_index('right')
                    tmp_tensor_2.axes_names[left_index] = 'L_2'
                    tmp_tensor_2.axes_names[right_index] = 'R_2'

                    input_contr = contract(tmp_tensor, tmp_tensor_2, contracted='d'+str(i))
                    right_contr = contract(input_contr, right_contr, 
                                           input_contr.ax_to_index(['right','R_2']), 
                                           right_contr.ax_to_index(['left','L_2']))
            else:
                right_contr = None


        # Compute d/dB of the L2 norm of the net by contracting the net only to one tensor B
        if (right_contr is not None) and (left_contr is not None):
            derivate = contract(B, right_contr, 'right', 'left')
            derivate = contract(left_contr, derivate, 'right', 'left')
        elif left_contr is None:
            derivate = contract(B, right_contr, 'right', 'left')
        elif right_contr is None:
            derivate = contract(left_contr, B, 'right', 'left')


        # Rename and swap left right index to obtain the correct shape of the derivate
        if left_dir == False:

            if self.l_pos == 0:
                left_index = derivate.ax_to_index('L_2')
                derivate.axes_names[left_index] = 'right'
            elif (self.l_pos > 0) and (self.l_pos < (self.N-2)):
                left_index = derivate.ax_to_index('L_2')
                right_index = derivate.ax_to_index('R_2')
                derivate.axes_names[left_index] = 'right'
                derivate.axes_names[right_index] = 'left'
            else: 
                right_index = derivate.ax_to_index('R_2')
                derivate.axes_names[right_index] = 'left'

        else:

            if self.l_pos == 1:
                left_index = derivate.ax_to_index('L_2')
                derivate.axes_names[left_index] = 'right'
            elif (self.l_pos > 1) and (self.l_pos < (self.N-1)):
                left_index = derivate.ax_to_index('L_2')
                right_index = derivate.ax_to_index('R_2')
                derivate.axes_names[left_index] = 'right'
                derivate.axes_names[right_index] = 'left'
            else: 
                right_index = derivate.ax_to_index('R_2')
                derivate.axes_names[right_index] = 'left'



        # Compute the loss contribution of the regularization
        indexes = B.axes_names
        loss_term = contract(B, derivate, 
                             B.ax_to_index(indexes), 
                             derivate.ax_to_index(indexes)).elem

        # Scale tensors elements with input constants
        derivate.elem = 2*weight_dec * derivate.elem
        loss_term = weight_dec * loss_term

        return loss_term, derivate
