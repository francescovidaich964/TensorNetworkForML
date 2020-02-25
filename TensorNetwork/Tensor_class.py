import numpy as np
import copy
from tqdm import tnrange
new = np.newaxis

class Tensor():
    """ 
    Class used to represent a Tensor.
    
    Attributes
    ----------
    elem: numpy array, optional
        Multi-dimensional matrix (ndarray) containing the elements of the tensor
    shape: tuple, optional
        Shape of the elem attribute
    axes_names: list of str, optional
        Names of each axis of the Tensor
    rank: int
        Rank of the tensor (i.e. length of the shape attribute)
    aggregations: dict of dict
        Each key is the name of an aggregated axis, whose values are the names of the 
        axes mapped in it and their original dimensions 
        
    Methods
    -------
    aggregate(axes_names=None, new_ax_name=None, debug=False)
        Maps multiple axes (axes_names) in a new one whose dimension is the product of the
        dimensions of the aggregated axis 
    disaggregate(ax)
        Unpack aggregated axis (ax) to original axes
 
    Notes
    -----
    If during initialization just the shape of the elements is provided, initializes the 
    elements randomly from the uniform distribution in [0,1] and then divides them by an
    heuristic factor sqrt(# elems/2) - KEEP IT UPDATED
    """
    
    def __init__(self, elem=None, shape=None, axes_names=None, scale=1.):
        """
        Parameters
        ----------
        elem: numpy array, optional
            Multi-dimensional matrix (ndarray) containing the elements of the tensor
        shape: tuple, optional
            Shape of the elem attribute
        axes_names: list of str, optional
            Names of each axis of the Tensor
        sigma: float, optional
            Not implemented
            
        Raises
        ------
        Exception
            If neither elem nor shape is provided
        ValueError
            If the number of names axes_names is different from the rank of the tensor
        TypeError
            If axes_names does not support the built-in len() function
        """
        # Numeric initialization
        if (elem is None) and (shape is not None):
            self.elem = np.random.random(size=shape) # uniform in [0,1]
            self.elem /= scale 
        elif elem is not None:
            self.elem = elem
        else:
            raise Exception('You have to provide either the elements of the tensor or its shape')
            
        # Relevant attributes initialization
        self.shape = self.elem.shape
        self.rank = len(self.shape)
        self.aggregations = {}

        if axes_names is not None:
            try:
                if len(axes_names) == self.rank:
                    # FIXME: history_axes_names is obsolete (check)
                    self.history_axes_names = [np.array(axes_names)] 
                    self.axes_names = np.array(axes_names)
                else:
                    raise ValueError("") # this error is handled with the except ValueError below
            except TypeError:
                print("=== Warning ===\nThe object that describes the indexes names have at least to support the built-in len function."\
                          +"\naxes_names attribute has not been inizialized.")
                self.axes_names = None
            except ValueError:
                print("=== Warning ===\nThe number of names should match the rank of the tensor."\
                          +"\naxes_names attribute has not been inizialized.")
                self.axes_names = None
        else:
            self.axes_names = None

        return


    def aggregate(self, axes_names=None, new_ax_name=None, debug=False):
        """ 
        Maps multiple axes (axes_names) in a new one whose dimension is the product of the
        dimensions of the aggregated axis 
        
        Parameters
        ----------
        axes_names: list of str, optional
            List containing the names of the axes to be aggregated
            (default is None and performs aggregation on all the axes)
        new_ax_name: str
            Name of the new axis (defautl is None, which raises a ValueError)
        debug: bool, optional
            If True, prints debugging information
            
        Raises
        ------
        ValueError
            If new_ax_name is not provided or if the Tensor has no names for the axes
        AssertionError
            If one of the names provided as input does not correspond to any axis' name 
        """
        
        dprint = print if debug else lambda *args, **kwargs : None
        
        # Sanity checks
        if (axes_names is None) and (new_ax_name is not None):
            axes_names = self.axes_names # if axes_names is None -> aggregate all axes
        elif new_ax_name is None:
            raise ValueError("You have to provide the name of the new axes")
            
        if self.axes_names is None:
            raise ValueError("This function can be called only if the axes names are defined")
            
        for name in axes_names:
            assert name in self.axes_names, "The " + name + " axes wasn't found in the tensor"
            
        dprint("Aggregating...")


        # Convert the axes names to their index positions
        indexes = self.ax_to_index(axes_names)
        
        # Store original shape of the aggregated indexes
        axes_sizes = np.array(self.shape)[indexes]
        self.aggregations[new_ax_name] = dict(zip(axes_names, axes_sizes))
        
        # Gather the non contracted indexes
        all_indexes = set(range(len(self.elem.shape)))
        other_indexes = list(all_indexes.difference(set(indexes)))
        other_indexes.sort()

        dprint("axes_numerical+other_axes: ", indexes+other_indexes)

        # Perform actual reshaping
        self.elem = np.transpose(self.elem, indexes+other_indexes)        
        other_sizes = np.array(self.shape)[other_indexes].tolist()
        self.elem = self.elem.reshape([-1]+other_sizes)
        
        # Update class members
        self.update_members(np.concatenate([[new_ax_name], self.axes_names[other_indexes]]))
        
        return
        

    def disaggregate(self, ax):
        """
        Unpack aggregated axis (ax) to original axes
        
        Parameters
        ----------
        ax: str
            Name of an aggregate axis to be disaggregated
           
        Raises
        ------
        AssertionError
            If the name provided does not correspond to an existing axis or to
            an axis that is not the result of an aggregation 
        """
        
        assert ax in self.axes_names, "The " + ax + " ax wasn't found in the tensor."
        assert ax in self.aggregations.keys(), "The " + ax + " does not represent an aggregated ax."
        
        original_dict = self.aggregations[ax]
        original_names = list(original_dict.keys())
        original_shape = list(original_dict.values())
        
        index = self.ax_to_index(ax)
        
        # transpose to have the aggregated index at the beginning
        permutation = [index] + np.arange(index).tolist() + np.arange(index+1, self.rank).tolist()
        self.elem = np.transpose(self.elem, permutation)
        self.update_members(self.axes_names[permutation])
        
        # Disaggregate axis by reshaping the tensor
        self.elem = self.elem.reshape(original_shape + list(self.shape[1:]))
        self.update_members(np.concatenate([original_names, self.axes_names[1:]]))
        
        # Remove aggregated index from the memory
        self.aggregations.pop(ax)
        
        return


    def transpose(self, permutation):
        """
        Changes the axes order according to the permutation of the names provided
        
        Parameters
        ----------
        permutation: list of str
            List containing the names of all the axes of the tensor in the order that
            must be obtained after the permutation
        """
        # permutation is axes_names in the new order
        indexes = self.ax_to_index(permutation)
        self.elem = np.transpose(self.elem, indexes)
        self.update_members(permutation)
        return


    def ax_to_index(self, axes):
        """
        Gets the indices associated to axes names given (axes)
        
        Parameters
        ----------
        axes: str or list of str
            Names of the axes of which we want to get the numerical
            positions (indices)
            
        Returns
        -------
        int or list of int
            Numerical indices of the axes
        """
        # handle single and multiple indices separately
        if type(axes) == str:
            return np.where(self.axes_names == axes)[0][0]
        else:
            return_axes = []
            for ax in axes:
                return_axes.append(np.where(self.axes_names == ax)[0][0])
            return return_axes


    def update_members(self, axes_names):
        """
        Updates axes names, shape and rank attributes after an aggregation or disaggregation
        
        Parameters
        ----------
        axes_names: list of str
            New names of the axes of the tensor
        """
        self.axes_names = np.array(axes_names)
        self.shape = self.elem.shape
        self.rank = len(self.shape)
        return
    
    
    def check_names(self):
        """
        Prints the type of axes_names attribute
        """
        print("="*10+"axes_names type"+"="*10)
        print(type(self.axes_names))
        

    def __str__(self):
        print("="*10+" Tensor description "+"="*10)
        print("Tensor shape: ", self.shape)
        print("Tensor rank: ", self.rank)
        print("Axes names: ", self.axes_names)
        return ""
    
    def __add__(self, o): 
        """
        Perform sum of two tensors permuting the axes of the second so that they are alligned.
        """

        # check all names match between two tensors
        assert np.all(np.isin(self.axes_names, o.axes_names)), "Error: axes don't match, cannot sum tensors."

        o.transpose(self.axes_names)
        t3 = self.elem + o.elem
        T3 = Tensor(elem = t3, axes_names = self.axes_names)
        return T3

