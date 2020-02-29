

    #########################################
    ############# OLD METHODS ###############
    #########################################

    def r_sweep(self, X, y, f, lr, weight_dec, var_hist=None):
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
            f = self.r_sweep_step(f, y, lr, batch_size, weight_dec, var_hist)
        return f
    

    def l_sweep(self, X, y, f, lr, weight_dec, var_hist=None):
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
            f = self.l_sweep_step(f, y, lr, batch_size, weight_dec, var_hist)
        return f
            

    def r_sweep_step(self, f, y, lr, batch_size, weight_dec, var_hist=None):
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
        MAE = np.abs(y-f.elem).mean()
        #print("Accuracy (before optim.): ", accuracy)
        #print("MAE (before optim.): ", MAE)
        ######################################################
        

        #f.elem = y-f.elem # overwrite f with (target - prediction)
        f_tmp = f.elem
        f.elem = y/f.elem
        deltaB = contract(f, phi, contracted="b")

        ### ORIGINAL ###
        # Swap left and right indixes 
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

        # Perform L2 regularization 
        L2_loss_term, L2_gradient = self.compute_L2_reg(B, weight_dec)
        deltaB -= L2_gradient


        ########## DEBUG ##########
        # Store history of B, deltaB, accuracy, output and MAE before the update:
        if var_hist is not None:
            var_hist[0].append(np.abs(B.elem).mean())
            var_hist[1].append(np.abs(deltaB.elem).mean())
            var_hist[2].append(accuracy)
            var_hist[3].append(np.abs(f_tmp).mean())
            var_hist[4].append(MAE)
            var_hist[5].append(L2_loss_term)
            var_hist[6].append(np.abs(L2_gradient.elem).mean())
        ###########################

        
        # gradient clipping -> rescale all elements of the gradient so that the
        # norm does not exceed the sum of the absolute values of B's entries
        B_measure = np.abs(B.elem).sum()
        if np.abs(deltaB.elem).sum() > B_measure:
            deltaB.elem /= np.abs(deltaB.elem).sum()/B_measure

        #deltaB.elem /= np.max(np.abs(deltaB.elem))
        deltaB.elem *= lr # multiply gradient for learning rate
        #print('DeltaB: ', np.abs(deltaB.elem).sum()) # debug

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
        MAE = np.abs(y-f.elem).mean()
        #print("Accuracy (after optim.): ", accuracy)
        #print("MAE (after optim.): ", MAE)
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
    

    def l_sweep_step(self, f, y, lr, batch_size, weight_dec, var_hist=None):
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
            phi = contract(phi, self.l_cum_contraction[-3], common = "b")
            
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
        MAE = np.abs(y-f.elem).mean()
        #print("Accuracy (before optim.): ", accuracy)
        #print("MAE (before optim.): ", MAE)
        ######################################################
        
        #print('f: ', np.abs(f.elem).sum())
        #f.elem = y-f.elem  # overwrite f with (target - prediction)
        f_tmp = f.elem
        f.elem = y/f.elem
        deltaB = contract(f, phi, contracted="b")


        ### ORIGINAL ###
        ## change left and right indices 
        #left_index = deltaB.ax_to_index('left')
        #right_index = deltaB.ax_to_index('right')
        #deltaB.axes_names[left_index] = 'right'
        #deltaB.axes_names[right_index] = 'left'
        if l==1:
            left_index = deltaB.ax_to_index('left')
            deltaB.axes_names[left_index] = 'right'
        elif (l > 1) and (l<(self.N-1)):
            left_index = deltaB.ax_to_index('left')
            right_index = deltaB.ax_to_index('right')
            deltaB.axes_names[left_index] = 'right'
            deltaB.axes_names[right_index] = 'left'
        else: 
            right_index = deltaB.ax_to_index('right')
            deltaB.axes_names[right_index] = 'left'

        # Perform L2 regularization 
        L2_loss_term, L2_gradient = self.compute_L2_reg(B, weight_dec, inverse=True)
        deltaB -= L2_gradient


        ########## DEBUG ##########
        # Store history of B, deltaB, accuracy, output and MAE before the update:
        if var_hist is not None:
            var_hist[0].append(np.abs(B.elem).mean())
            var_hist[1].append(np.abs(deltaB.elem).mean())
            var_hist[2].append(accuracy)
            var_hist[3].append(np.abs(f_tmp).mean())
            var_hist[4].append(MAE)
            var_hist[5].append(L2_loss_term)
            var_hist[6].append(np.abs(L2_gradient.elem).mean())
        ###########################


        # gradient clipping -> rescale all elements of the gradient so that the
        # norm does not exceed the sum of the absolute values of B's entries
        B_measure = np.abs(B.elem).sum()
        if np.abs(deltaB.elem).sum() > B_measure:
            deltaB.elem /= np.abs(deltaB.elem).sum()/B_measure    

        #deltaB.elem /= np.max(np.abs(deltaB.elem))
        deltaB.elem *= lr # multiply gradient for learning rate
        #print('DeltaB: ', np.abs(deltaB.elem).sum()) # debug

        #print('B: \t', np.abs(B.elem).sum()), 
        #print('deltaB: ', np.abs(deltaB.elem).sum())

        # update B    
        B = B + deltaB
            
        # compute new output of the net (out is like f, but with new A weights)
        out = contract(B, self.TX[l-1], contracted='d'+str(l-1))
        out = contract(out, self.TX[l], contracted='d'+str(l), common='b')
        
        if l == self.N-1:
            # no right term
            out = contract(self.l_cum_contraction[-3], out, 'right', 'left', common = "b") # ok
        
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
        MAE = np.abs(y-f.elem).mean()
        #print("Accuracy (after optim.): ", accuracy)
        #print("MAE (after optim.): ", MAE)
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
    