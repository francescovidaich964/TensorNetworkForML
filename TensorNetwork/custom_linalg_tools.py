
import numpy as np
import copy
from tqdm import tnrange
new = np.newaxis

from Tensor_class import Tensor


def _contract_(T1, T2, contracted_axis1, contracted_axis2, common_axis1=[], common_axis2=[]):
    """
    Contracts two tensors along one axis.
    
    Parameters
    ----------
    T1, T2: Tensor
        Tensors to be contracted
    contracted_axis1, contracted_axis2: list of int
        Indices of the axes of T1 and T2 to be contracted
    common_axis1, common_axis2: list of int, optional
         Indices of the axes of T1 and T2 in common
    
    Returns
    -------
    Tensor
        Contracted tensors
        
    Raises
    ------
    AssertionError
        If the number of common axes is different or if the dimensions of contracted or common axes 
        do not match between tensors
    """

    # Sanity checks
    assert len(common_axis1) == len(common_axis2), "number of common axes is different"
    
    if type(contracted_axis1) != list:
        # assuming contracted_axis1/2 is numeric
        assert T1.shape[contracted_axis1] == T2.shape[contracted_axis2], "dimensions of contracted axes do not match"
        contracted_axis1 = [contracted_axis1]
        contracted_axis2 = [contracted_axis2]
    
    for i in range(len(common_axis1)):
        # assuming common_axis1/2 is numeric
        assert T1.shape[common_axis1[i]] == T2.shape[common_axis2[i]], "dimensions of common axes do not match"
        
    original_shape1 = np.array(T1.shape)
    original_shape2 = np.array(T2.shape)
        
    def perm(contracted_axis, original_shape, common_axis):
        # assuming contracted_axis and common_axis list of integers
        # astype is for handling the case in the first array is empty, 
        # in which the function cannot infer the type
        last_axis = np.concatenate((common_axis, contracted_axis)).astype("int64")         

        remaining_axis = np.delete(np.arange(len(original_shape)), last_axis)
        permutation = np.concatenate((remaining_axis, last_axis))
        return permutation

    permutation1 = perm(contracted_axis1, original_shape1, common_axis1)
    permutation2 = perm(contracted_axis2, original_shape2, common_axis2)

    shape1 = original_shape1[permutation1]
    shape2 = original_shape2[permutation2]

    # param for match the rank of the two shapes
    unique1 = len(shape1)-len(common_axis1)-len(contracted_axis1)
    unique2 = len(shape2)-len(common_axis1)-len(contracted_axis1)

    new_shape1 = np.concatenate((shape1[:unique1],[1 for i in range(unique2)],shape1[unique1:])).astype("int64")
    new_shape2 = np.concatenate(([1 for i in range(unique1)],shape2)).astype("int64")
    
    T1.transpose(T1.axes_names[permutation1])
    T2.transpose(T2.axes_names[permutation2])
    
    T3_axes_names = np.concatenate([T1.axes_names[:unique1], T2.axes_names[:T2.rank-len(contracted_axis2)]])
    #else: 
    #    T3_axes_names = None

    T3 = (T1.elem.reshape(new_shape1)*T2.elem.reshape(new_shape2))
    if len(contracted_axis1) > 0:
        # if len(contracted_axis1) == 0 just to tensor product
        T3 = T3.sum(axis=-1)
        
    T3 = Tensor(elem=T3, axes_names=T3_axes_names)
    return T3


def contract(T1, T2, contracted_axis1=[], contracted_axis2=[], common_axis1=[], common_axis2=[], contracted=None, common=None):
    """
    Contracts two tensors along one axis.
    
    Parameters
    ----------
    T1, T2: Tensor
        Tensors to be contracted
    contracted_axis1, contracted_axis2: str, optional
        Names of the axes of T1 and T2 to be contracted
    common_axis1, common_axis2: str or list of str, optional
        Names of the axes of T1 and T2 in common
    contracted: str, optional
        Shortcut for contracted_axis1/2 if the names are the same
    common: str or list of str, optional
        Shortcut for common_axis1/2 if the names are the same
    
    Returns
    -------
    Tensor
        Contracted tensors
        
    Notes
    -----
    This function is a wrapper for the _contract_ function, where the actual contracion is performed
    (see help(tn._contract_) for more info)
    
    TODO: check that contracted_axis1 and 2 are strings and not lists
    """
    
    # if contracted is specified, assign its value to both contracted axes
    if contracted is not None:
        contracted_axis1 = contracted
        contracted_axis2 = contracted
 
    # if common is specified, assign its value to both common axes
    if common is not None:
        common_axis1 = common
        common_axis2 = common
        
    # If common_axis1/2 are provided as int or str, redefine them as lists
    if type(common_axis1) != list:
        common_axis1 = [common_axis1]
    if type(common_axis2) != list:
        common_axis2 = [common_axis2]

    # If contracted_axis1/2 is a string, get the corresponding numerical axis
    if type(contracted_axis1) == str:
        contracted_axis1 = T1.ax_to_index(contracted_axis1)
    if type(contracted_axis2) == str:
        contracted_axis2 = T2.ax_to_index(contracted_axis2)
       
    # If common_axis1/2 is a list of strings, get the corresponding numerical axes
    temp = []
    for key in common_axis1:
        if type(key) == str:
            temp.append(np.where(T1.axes_names == key)[0][0])
        else:
            temp.append(key)
    common_axis1 = temp
    # should also work something like common_axis1 = T1.ax_to_index(common_axis1)

    temp = []
    for key in common_axis2:
        if type(key) == str:
            temp.append(np.where(T2.axes_names == key)[0][0])
        else:
            temp.append(key)
    common_axis2 = temp
    
    # call _contract_ function for actual contraction
    return _contract_(T1, T2, contracted_axis1, contracted_axis2, common_axis1, common_axis2)


def partial_trace(T, ax1, ax2):
    """
    Compute the partial trace of a tensor (contraction of two axis between themselves)
    
    Parameters
    ----------
    T: Tensor
        Tensor on which partial trace is performed
    ax1, ax2: str
        Names of the two axes to be traced out
        
    Returns
    -------
    Tensor
        Tensor obtained through partial trace
        
    TODO: write assertion to see if dimensions of ax1 and 2 match
    """
    traced_axes = np.array([ax1,ax2])
    traced_indexes = T.ax_to_index(traced_axes)
    remaining_axis = np.delete(T.axes_names, traced_indexes)
    permutation = np.concatenate((traced_axes, remaining_axis))
    T.transpose(permutation)
    t = T.elem.trace(axis1=0, axis2=1)
    T = Tensor(elem=t, axes_names=remaining_axis)
    return T

