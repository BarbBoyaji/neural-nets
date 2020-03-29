#generic cnn class here
import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

class GenericConv(object):
    """
    An n-layer convolutional network with configurable architecture:
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
  
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=[100,100], num_classes=10, weight_scale=1e-3, reg=0.9,
               dtype=np.float32, optim='adagrad', 
               pool_params = {'pH': 2, 'pW': 2, 's': 2}, 
               architecture = {"CONV_R": 3, "CONV_R_P": 1, "AFFINE": 3, "SOFTMAX": 1 },
               use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in all convolutional layers
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - optim: the optimizer to use
        - pool_params: information about the pooling layers
        - architecture: possible keys are CONV (convolutional), R (ReLU), P (Pooling), CONV_R (convolution and ReLU), CONV_R_P (convolution and ReLU and pooling), AFFINE (fully connected), SOFTMAX, or SVM. Each key corresponds to an integer value that indicates the number of times that layer should repeat.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.architecture = architecture
        
        #unpack pooling parameters
        pH = pool_params['pH']
        pW = pool_params['pW']
        s = pool_params['s']
        
        #grab input dims
        C, H, W = input_dim
        
        #parameters that keep track of the height and width of last hidden layers of CNN
        H_ = H #(H-filter_size)//1 + 1
        W_ = W #(W-filter_size)//1 + 1  
        
        #index that keeps track of weights/biases during initialization
        i=0
        
        #index that keeps track of the hidden layers
        hD=0
        
        #index that is the last layer
        end = 0
        for _, l in architecture.items():
            end +=l
        print(f"end is: {end}")
        
        for layer, num in architecture.items():
           
            print(f"{layer} at i={i}")
            
            #convolutional layer
            if (layer == "CONV") | (layer == "CONV_R"):
                
                for k in range(num):
                    print(f"\t{k}")
                    if i == 0:
                        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
                        self.params['b1'] = np.zeros(num_filters,)
                        i+=1
                        print(f"\tW: {self.params['W1'].shape}")
                        print(f"\tb: {self.params['b1'].shape}")                       
                    else:
                        ws = 'W' + str(i+1)
                        bs = 'b' + str(i+1)
                        self.params[ws] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
                        self.params[bs] = np.zeros(num_filters,)
                        i+=1
                        print(f"\tW: {self.params[ws].shape}")
                        print(f"\tb: {self.params[bs].shape}")
                
            elif layer == "R":
                if i == 0:
                    print("SHOULD NOT HAVE RELU AS FIRST LAYER!")
                    return
                
            elif layer == "P":
                H_ = (H-pool_params['pH'])//pool_params['s'] + 1
                W_ = (W-pool_params['pW'])//pool_params['s'] + 1
                
            elif layer == "CONV_R_P":
                for k in range(num):
                    print(f"\t{k}")
                    if i == 0:
                        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
                        self.params['b1'] = np.zeros(num_filters,)
                        i+=1
                        
                        H_ = (H_-filter_size)//1 + 1
                        W_ = (W_-filter_size)//1 + 1
                        
                        print(f"\tW: {self.params['W1'].shape}")
                        print(f"\tb: {self.params['b1'].shape}")
                    elif i == end:
                        print("SHOULD NOT HAVE CONVOLUTIONAL AS LAST LAYER!")
                        return
                    else:
                        ws = 'W' + str(i+1)
                        bs = 'b' + str(i+1)
                        self.params[ws] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
                        self.params[bs] = np.zeros(num_filters,)
                        i+=1
                        print(f"\tW: {self.params[ws].shape}")
                        print(f"\tb: {self.params[bs].shape}")
                        
                        H_ = (H_-filter_size)//1 + 1
                        W_ = (W_-filter_size)//1 + 1
                    
            elif layer == "AFFINE":
                for k in range(num):
                    print(f"\t{k}")
                    if k ==0:
                        ws = 'W' + str(i+1)
                        bs = 'b' + str(i+1)
                        self.params[ws] = np.random.normal(0, weight_scale, (H_*W_*num_filters, num_classes))
                        self.params[bs] = np.zeros(num_classes,)
                        print(f"\tW: {self.params[ws].shape}")
                        print(f"\tb: {self.params[bs].shape}")
                    
                    elif i == end:
                        ws = 'W' + str(i+1)
                        bs = 'b' + str(i+1)
                        self.params[ws] = np.random.normal(0, weight_scale, (hidden_dim[hD], num_classes))
                        self.params[bs] = np.zeros(num_classes,)
                        print(f"\tW: {self.params[ws].shape}")
                        print(f"\tb: {self.params[bs].shape}")
                    else:
                        ws = 'W' + str(i+1)
                        bs = 'b' + str(i+1)
                        self.params[ws] = np.random.normal(0, weight_scale, (hidden_dim[hD-1], hidden_dim[hD]))
                        self.params[bs] = np.zeros(hidden_dim[hD],)
                        hD +=1
                        print(f"\tW: {self.params[ws].shape}")
                        print(f"\tb: {self.params[bs].shape}")
            else:
                print(f"UNDEFINED LAYER: {layer}")
                

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        for k, v in self.params.items():
              self.params[k] = v.astype(dtype)
     
 
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the three layer CNN.  Store the output
        #   scores as the variable "scores".
        # ================================================================ #
        
        #using API in conv_layer_utils
        #do convolution, relu, and pool
        
        #index that keeps track of layer
        i=0
        #index that is the last layer
        end = 0
        for _, l in self.architecture.items():
            end +=l
        print(f"end is: {end}")
        
        cache_conv = []
        cache_conv_r = []
        cache_aff = []
        cache_relu = []
        
        for layer, num in self.architecture.items():
           
            print(f"{layer} at i={i}")
            
            #convolutional layer
            if (layer == "CONV"): 
                for k in range(num):
                    print(f"\t{k}")
                    
                    ws = 'W' + str(i+1)
                    bs = 'b' + str(i+1)
                    X, cache = conv_forward(X, self.params[ws], self.params[bs], conv_param)
                    cache_conv.append(cache_c)
                    i+=1
                    
            elif (layer == "CONV_R"):
                for k in range(num):
                    print(f"\t{k}")   
                    
                    ws = 'W' + str(i+1)
                    bs = 'b' + str(i+1)
                    X, cache = conv_relu_forward(X, self.params[ws], self.params[bs], conv_param)
                    cache_conv_r.append(cache)
                    i+=1
                    
            elif (layer == "CONV_R_P"):
                for k in range(num):
                    print(f"\t{k}")   
                    
                    ws = 'W' + str(i+1)
                    bs = 'b' + str(i+1)
                    X, cache = conv_relu_pool_forward(X, self.params[ws], self.params[bs], conv_param, pool_param)
                    cache_conv_r.append(cache)
                    i+=1
            elif (layer == "AFFINE"):
                for k in range(num):
                    print(f"\t{k}") 
                    
                    ws = 'W' + str(i+1)
                    bs = 'b' + str(i+1)
                    X, cache = affine_forward(X, self.params[ws], self.params[bs])
                    cache_aff.append(cache)
                    i+=1
            elif (layer == "R"):
                X, cache_relu1 = relu_forward(X)
                cache_relu.append(cache)
        
        
        #softmax?

        scores = X
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        if y is None:
            return scores
        
        loss, grads = 0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backward pass of the three layer CNN.  Store the grads
        #   in the grads dictionary, exactly as before (i.e., the gradient of 
        #   self.params[k] will be grads[k]).  Store the loss as "loss", and
        #   don't forget to add regularization on ALL weight matrices.
        # ================================================================ #
        #conv - relu - 2x2 max pool - affine - relu - affine - softmax
        loss, dout = softmax_loss(scores, y)
        
        dx3, dw3, db3 = affine_backward(dout, cache_aff2)
        
        dx2relu = relu_backward(dx3, cache_relu1)
        
        dx2, dw2, db2 = affine_backward(dx2relu, cache_aff1)
        
        dx1, dw1, db1 = conv_relu_pool_backward(dx2, cache_conv)
        
        grads = {'W1': dw1, 'W2': dw2, 'W3': dw3,
                 'b1': db1, 'b2': db2, 'b3': db3}

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grads
  