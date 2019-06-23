import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from PIL import Image
from scipy import ndimage
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward, load_data, print_mislabeled_images
import pickle
import skimage
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (10,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = -np.sum(Y*np.log(AL) + (1 - Y) * np.log(1 - AL))/m
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, cache[0].T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0])
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0])
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def predict(X, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((10,m))
    ind = np.zeros((1,m))
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    for i in range(m):
        ind[:, i] = np.unravel_index(np.argmax(probas[:, i], axis  = None), probas[:, i].shape)[0]
        p[np.unravel_index(np.argmax(probas[:, i], axis  = None), probas[:, i].shape)[0]][i] = 1 
    # convert probas to 0/1 predictions
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
        
    return p,ind 

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False,parameters = None):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    print(layers_dims)
    costs = []                         # keep track of cost
    para_list = []
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    if parameters == None:
        parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        if i % 100 == 0:
            para_list.append(parameters)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters, para_list

file1 = open("dataset.mm", "rb+")
(X_train, Y_train), (X_test, Y_test) = pickle.load(file1)
file1.close()

parameters = None
par_lis = []
dim_lis = []
while True:
    print("Enter your options")
    print("Enter 1 to import model from file")
    print("Enter 2 to train model")
    print("Enter 3 to view result on test image")
    print("Enter 4 to view result on custom image")
    print("Enter 5 to further train same Model")
    try:
        i = int(input("Enter your options:"))
    except:
        print("Enter proper input!!!")
        continue
    if i == 1:
        try:
            f = input("Enter filename: ")
            file1 = open(f, "rb+")
            parameters, dim_lis = pickle.load(file1)
            file1.close()
        except:
            print("Error Opening file : Check filename")
            continue
    if i ==2:
        ls = []
        len_l = int(input("Enter number of Layers: "))
        ls.append(28*28)
        print("Frist and Last layer will be by default 28*28 and 10 respectively:")
        for i in range(1, len_l -1):
            s = int(input("Enter size of layer {}:".format(i)))
            ls.append(s)
        ls.append(10)
        dim_lis = ls
        ni = int(input("Enter number of iterations: "))
        lr = float(input("Enter Learning rate: "))
        limit = list(map(int,input("Enter index [from-to]").split("-")))
        parameters, par_lis = L_layer_model(X_train[:,limit[0]:limit[1]], Y_train[:, limit[0]: limit[1]], layers_dims = ls, learning_rate = lr, num_iterations = ni, print_cost=True,parameters = None)
        name = input("Enter model name:")
        file = open(name+".mod", "wb+")
        pickle.dump([parameters, ls], file)
        file.close()
        
        
    elif i == 3:
        if parameters == None:
            print("Please select some model!!")
            continue
        pp, ind = predict(X_test, parameters)
        ll = int(input("Enter number of images you want to see:"))
        k = 1
        for i in range(ll):
                im = plt.subplot(4,6,k)
                im.get_xaxis().set_visible(False)
                im.get_yaxis().set_visible(False)
                X = X_test[:, i]
                Y = X_test[:, i]
                plt.imshow(X.reshape(28, 28))
                plt.title("Prediction : {}".format(ind[0][i]))
                if k == 24 or i == ll - 1:
                    plt.show()
                    k = 0
                k+=1
    elif i == 4:
        if parameters == None:
            print("Please select some model!!")
            continue
        while True:
            my_image = input("Enter you Image name inside 'images/' directory (or -1 to exit):")
            if my_image == "-1":
                break
            fname = "images/" + my_image
            #image = np.array(ndimage.imread(fname, flatten=False))
            image = np.array(matplotlib.pyplot.imread(fname))
            print(image.shape)
            image = 255 - image
            #my_image = scipy.misc.imresize(image, size=(28,28)).reshape((1, 28*28)).T
            my_image = skimage.transform.resize(image,(28,28, 3))
            my_image = np.sum(my_image, axis = 2)/3
            my_image = my_image.reshape(1, 28*28).T
            
            pp, ind = predict(my_image, parameters)
            plt.imshow(image)
            plt.title("Prediction : {}".format(ind[0][0]))
            plt.show()
    elif i == 5:
        if parameters == None or dim_lis == []:
            print("please select proper model")
            continue
        ni = int(input("Enter number of terations:"))
        lr = float(input("Enter learning rate:"))
        limit = list(map(int,input("Enter index [from-to]").split("-")))
        parameters, par_lis = L_layer_model(X_train[:, limit[0]:limit[1]], Y_train[:, limit[0]:limit[1]], layers_dims = dim_lis, learning_rate = lr, num_iterations = ni, print_cost=True,parameters = parameters)
        nm = input("Enter name of the model:")
        file1 = open(nm+".mod", "wb+")
        pickle.dump([parameters, dim_lis], file1)
        file1.close()
        
        
        









##        image = image/255.
##        my_image = skimage.transform.resize(image,(num_px,num_px)).reshape((1, num_px*num_px*3)).T
##        my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
##        my_predicted_image = predict(d["w"], d["b"], my_image)
##
##        plt.imshow(image)
##        
##        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
##        plt.show()
##
##file = open("Trialmodel.mod", "rb+")
##
##parameters = pickle.load(file)
##file.close()
##
##pp = predict(X_test, Y_test, parameters)
##for i in range(10000):
##    X = X_test[:, i]
##    Y = X_test[:, i]
##    ind = np.unravel_index(np.argmax(pp[:, i], axis=None), pp[:, i].shape)[0]
##    print(ind)
##    plt.imshow(X.reshape(28, 28))
##    plt.show()
####parameters, P_list =  L_layer_model(X_train[:,0:10000], Y_train[:, 0:10000], [784, 30, 10, 10], learning_rate = 0.0075, num_iterations = 1400, print_cost=True)
##
##
##
