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
        
        
        self.l_cum_contraction = [] # init left cumulative contraction array
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
        
        self.r_cum_contraction = [] # init right cumulative contraction array
        # sweep from right to left
        for i in range(self.N-1):
            #print("\nleft sweep step ",self.N-1-i)
            f = self.l_sweep_step(f, y, lr, batch_size)
        
        return f
