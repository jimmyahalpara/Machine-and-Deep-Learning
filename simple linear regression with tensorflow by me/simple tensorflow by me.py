#importing important modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01 #learning rate for optimizer
train_epoches = 100 

x_train = np.linspace(-1, 1, 101) 
y_train = 2*x_train + np.random.randn(*x_train.shape)*0.33 


#setting up nodes as placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#multiplying X and w which is our linear regression model
def model(X, w):
    return tf.multiply(X,w)

w = tf.Variable(0.0, name = "weightskkk") #creating variable for weights
y_model = model(X, w) #predicting
cost = tf.square(Y - y_model) #cost function for our model

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #our Gradient descent optimizer

w_val = None
with tf.Session() as sess:
    init = tf.global_variables_initializer() #initializing variables
    sess.run(init) #same as above
    for epoch in range(train_epoches): #traning model train_epoches times which is currently 100
        for (x,y) in zip(x_train, y_train): #extracting x,y as pairs
    ##        writer = tf.summary.FileWriter("output", sess.graph) #writing summery for tensorboard 
            sess.run(train_op, feed_dict = {X: x, Y:y}) #running optimizer by feeding the values of placeholders X, and Y
    ##        writer.close()
    w_val = sess.run(w) # extracting trained weights

plt.scatter(x_train,y_train) # creating graph
y_learned = x_train*w_val
plt.plot(x_train, y_learned)
plt.show()

