#from Deeplearning.ai course 1 Assignment
#modifications by Jimmy Kumar Ahalpara
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

def layer_sizes(X,Y, n_h = 4):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
    W1 -- weight matrix of shape (n_h, n_x)
    b1 -- bias vector of shape (n_h, 1)
    W2 -- weight matrix of shape (n_y, n_h)
    b2 -- bias vector of shape (n_y, 1)
    """
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    ##assert(A2.shape == (1, X.shape[1]))##hard coded for our dataset
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] 
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs)/m
    
    cost = np.squeeze(cost)      
    assert(isinstance(cost, float))
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2) , 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    #np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

while True:
    print("Select the data you want:")
    print("0. Flower \n1. Noisy Circles \n2. Noisy Moons \n3. Blobs \n4. Gaussian Quantiles\n5. Testing with differend hidden neurons")
    i = int(input("Enter no: "))
    noisy_circles, noisy_moons, blobs, gaussian_quantiles = load_extra_datasets()
    if i != 5:
        datasets = [load_planar_dataset(), noisy_circles, noisy_moons, blobs, gaussian_quantiles]
        X, Y = datasets[i]
    
        if i >0:
            X, Y = X.T, Y.reshape(1, Y.shape[0])
        if i == 3:
            Y = Y%2
        plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
        plt.show()
        n_h = int(input("Enter number of hidden layers: "))
        num_iterations = int(input("Enter number of iterations"))
        parameters = nn_model(X, Y, n_h, num_iterations, print_cost=True)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        plt.title("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
        plt.show()
    else:
        print("Starting Testing with 1,2,3,4,20 Hidden Neurons")
        learning_rate = float(input("Enter Learning Rate:"))
        num_iterations = int(input("Enter number of iterations: "))
        p = {"Flower":load_planar_dataset(),
             "NoisyCircle":noisy_circles,
             "NoisyMoons":noisy_moons,
             "Blobs":blobs,
             "GaussianQuantiles":gaussian_quantiles
            }
        for k in p:
            X, Y = p[k]
            if k != "Flower":
                X, Y = X.T, Y.reshape(1, Y.shape[0])
            if k == "Blobs":
                Y = Y%2
            plt.figure(figsize=(20, 30))
            hidden_layer_sizes = [1, 2, 3, 4, 5, 20]
            for i, n_h in enumerate(hidden_layer_sizes):
                plt.subplot(2, 3, i+1)
                parameters = nn_model(X, Y, n_h, num_iterations)
                predictions = predict(parameters, X)
                accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
                plt.title('No.H.L. Neurons: {},     Accuracy: {}%'.format(n_h, accuracy))
                plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
                print ("Model {} hidden units: {} Total Iterations {} Accuracy {}%".format(k,n_h,5000, accuracy))
            plt.show()
            
        
        
##parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
##plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
##plt.title("Decision Boundary for hidden layer size " + str(4))


##plt.figure(figsize=(16, 32))
##hidden_layer_sizes = [1, 2, 3, 4, 5, 20]
##for i, n_h in enumerate(hidden_layer_sizes):
##    plt.subplot(5, 2, i+1)
##    plt.title('Hidden Layer of size %d' % n_h)
##    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
##    plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
##    predictions = predict(parameters, X)
##    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
##    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

### Datasets
##noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
##
##datasets = {"noisy_circles": noisy_circles,
##            "noisy_moons": noisy_moons,
##            "blobs": blobs,
##            "gaussian_quantiles": gaussian_quantiles}
##
##
##dataset = "gaussian_quantiles"
##
##X, Y = datasets[dataset]
##X, Y = X.T, Y.reshape(1, Y.shape[0])
##
### make blobs binary
##if dataset == "blobs":
##    Y = Y%2
##
### Visualize the data
##plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
