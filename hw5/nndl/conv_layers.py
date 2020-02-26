import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param['pad']
    stride = conv_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of a convolutional neural network.
    #   Store the output as 'out'.
    #   Hint: to pad the array, you can use the function np.pad.
    # ================================================================ #

    #get shapes
    N, C, H, W = x.shape
    F, _, hh, ww = w.shape

    H_ = int(1 + (H + 2 * pad - hh) / stride)
    W_ = int(1 + (W + 2 * pad - ww) / stride)
    
    #sanity check
    #print(f"N: {N}, F: {F}\nH: {H}, W: {W}\nH_: {H_}, W_: {W_}\nhh: {hh}, ww: {ww}")
    
    #pad the array
    if pad != None:
        x = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
    
    #convolve
    out = np.zeros(shape=(N, F, H_, W_))

    for n in range(N):
        for f in range(F): 
            for i in range(H_):
                for j in range(W_):
                    #print(f"n = {n}, f = {f}, i = {i}, j = {j}")
                    sub = x[n, :, i*stride:(i*stride+hh), j*stride:(j*stride+ww)]             
                    intermed = np.sum(sub * w[f])+ b[f] #hadamard product, summed, added to biases of each class
                    out[n, f, i, j] = intermed
            
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    N, F, out_height, out_width = dout.shape
    x, w, b, conv_param = cache

    stride, pad = [conv_param['stride'], conv_param['pad']]
    xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    num_filts, _, f_height, f_width = w.shape

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of a convolutional neural network.
    #   Calculate the gradients: dx, dw, and db.
    # ================================================================ #
    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    db = np.zeros_like(b)
    
    #print(f"shape of dout: {dout.shape}")
    #print(f"shape of dx: {dx.shape}")
    #print(f"shape of dw: {dw.shape}")
    #print(f"shape of db: {db.shape}")
    
    for n in range(N):
        for f in range(F):
            for i in range(out_height):
                for j in range(out_width):
                    #print(f"n = {n}, f = {f}, i = {i}, j = {j}")
                    dx[n, :, i*stride:(i*stride+f_height), j*stride:(j*stride+f_width)] += w[f]*dout[n,f,i,j]
                    dw[f] += x[n, :, i*stride:(i*stride+f_height), j*stride:(j*stride+f_width)]*dout[n,f,i,j]
    db = np.sum(dout, axis=(0,2,3))
    dx = dx[:, :, pad:-pad, pad:-pad]
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling forward pass.
    # ================================================================ #
    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    s = pool_param['stride']
    
    N,C,H,W = x.shape

    H_ = int((H - pH) / s + 1)
    W_ = int((W - pW) / s + 1)
    
    #sanity check
    #print(f"N: {C}, F: {C}\nH: {H}, W: {W}\nH_: {H_}, W_: {W_}\npH: {pH}, pW: {pW}\nstride: {s}")
    
    #initialize out
    out = np.zeros( (N,C,H_,W_) )
    
    for i in range(H_):
        for j in range(W_):
            sub = x[:, :, i*s:(i*s+pH), j*s:(j*s+pW)]
            #print(f"shape of sub: {sub.shape}")
            out[:, :, i, j] = (sub).max(axis=(2,3))
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling backward pass.
    # ================================================================ #
    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    s = pool_param['stride']
    
    N,C,H,W = x.shape

    H_ = int((H - pH) / s + 1)
    W_ = int((W - pW) / s + 1)
    
    dx=np.zeros(x.shape)

    for n in range(N):
        for c in range(C):
            for i in range(H_):
                for j in range(W_):
                    sub = x[n,c,i*s:i*s+pH,j*s:j*s+pW]
                    max_ind1, max_ind2 = np.unravel_index(np.argmax(sub), (pH, pW))
                    dmax = np.zeros((pH,pW))
                    dmax[max_ind1, max_ind2] = dout[n,c,i,j]
                    #print(f"dmax:\n{dmax}")
                    dx[n,c,i*s:i*s+pH,j*s:j*s+pW] = dmax

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm forward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #
    
    N,C,H,W = x.shape
    
    #initialize the data
    data = np.zeros_like( (C, N, H, W))
    data = x.swapaxes(0,1)
    data = data.reshape(C, N*H*W)
    
    #feed data into batchnorm forward
    out, cache = batchnorm_forward(data.T, gamma, beta, bn_param)
    
    #reshape (undo what we did)
    out = out.T.reshape(C, N, H, W)
    out = out.swapaxes(0,1)


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm backward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #
    
    N,C,H,W = dout.shape
    
    #initialize the data
    deriv = np.zeros_like( (C, N, H, W))
    deriv = dout.swapaxes(0,1)
    deriv = deriv.reshape(C, N*H*W)
    
    #feed data into batchnorm backward
    dx, dgamma, dbeta = batchnorm_backward(deriv.T, cache)

    #reshape the dx value
    dx = dx.T.reshape(C, N, H, W).swapaxes(0,1)
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx, dgamma, dbeta