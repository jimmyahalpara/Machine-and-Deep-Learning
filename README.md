# **Machine Learning and Deep Learning**
## My Work 
* [Handwriting Classification Using Deep Neural Network](https://github.com/jimmyahalpara/Machine-and-Deep-Learning/tree/master/Handwriting%20Classification%20using%20Deep%20Neural%20Network)
	* Implemented Deep Neural Network from previous project **Cat Classification Using Deep Neural Network**
	* Modified Input and Output layer size
	* Used **MNIST** dataset
	* Added Feature to Save trained model and retrain same model
	* Used modules **Numpy, Matplotlib, Scipy, PIL, pickle, Skimage**
* [Cat Classification Using Deep Neural Network](https://github.com/jimmyahalpara/Machine-and-Deep-Learning/tree/master/Cat%20Classification%20with%20Deep%20Neural%20Network)
	* Implemented Neural Network with Multiple Layers
	* Added feature for saving our trained model
	* Can predict from images provided from user
	* Created by doing some modification in deeplearning.ai assignment
	* Used modules **Numpy, Scipy, Matplotlib, PIL, pickle**
* [Classification with Shallow Neural Network](https://github.com/jimmyahalpara/Machine-and-Deep-Learning/tree/master/Classification%20using%20Shallow%20Neural%20Network)
	* Implemented Neural network with one Hidden Layer
	* Trained using dataset from **sklearn**
	* Compared Accuracy against different **Datasets**, **No. of Hidden Layer Neurons**
	* Used modules **Sklearn, Matplotlib, Numpy**
* [Logistic Regression - Cat Classification](https://github.com/jimmyahalpara/Machine-and-Deep-Learning/tree/master/Cat%20Classification%20with%20Logistic%20Regression)
	* Made by keeping neural network in mind.
	* Compared accuracy for different **Learning Rate** and **No of Iterations**
	* created by doing some modification in my Deeplearning.ai programming assignment
	* Used modules **Numpy, Matplotlib, Scipy, Skimage, Pickle, PIL**
* [Linear Regression with Tensorflow](https://github.com/jimmyahalpara/Machine-and-Deep-Learning/tree/master/simple%20linear%20regression%20with%20tensorflow%20by%20me)
	* Used Tensorflow
	* Used Gradient Descent for optimization
	* Random Dataset created using Numpy
	* Used modules **Tensorflow**,**Numpy** and **Matplotlib**. 


Here I'll be posting my work as I learn and make.

## Resources from where I am learning
1. [Machine Learning by Stanford University- Coursera](https://www.coursera.org/learn/machine-learning)
2. [Deeplearnin.ai Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
	* [Neural Network and Deeplearning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning)
	* [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)
	* [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects?specialization=deep-learning)
	* [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)
	* [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models)
3. [Tensorflow Documentation](https://www.tensorflow.org/learn)
4. [Scipy and Numpy Documentation](https://docs.scipy.org/doc/)
### 1.  My first program of linear regression 

My first program of linear regression was just plotting the hypothesis on the datasets randomly created using **numpy** and plotted using **matplotlib** in **python** and model trainded by **tensorflow** using **gradient descent algorithem**.

![plotting of hypothesis](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/simple%20linear%20regression%20with%20tensorflow%20by%20me/Capture.PNG)

#### How linear regression and gradient descent works

![model training](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/simple%20linear%20regression%20with%20tensorflow%20by%20me/Webp.net-gifmaker.gif)

Let's say you have three points with following positions

![linear regression example](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/Guide%20materials/linear%20regression%201.png)

We want to make prediction on this we wan't a line that traces all the points and to the prediction will be easy in the future.
But we need the line such that it passes all the points roughly or maybe we can minimize the error between the point and line that will make our line approximatly pass over our points.
Lets say in this diagram the distance between two points on X any Y axis represent one unit distance.
Tracing the points perfectly we get the following line that we want from regression.

![tracing](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/Guide%20materials/linear%20regression%202.png)

In this figure we want eqution of the pink line such that error is minimized.

Lets initialize the line with following eqution

**y = x<sub>1</sub>.x + x<sub>0</sub>** where **x<sub>1</sub>** is the slope of the line and **x<sub>0</sub>** is intercept of the line.
We have to adjust the slpe and the intercept of the line such that the error is minimized.
lets say we have randomly initialized the slope and intercept as follows **x<sub>1</sub> = 0** and **x<sub>0</sub> = 2** let's not worry about the error and tracing at the point of time. Our eqution will become **y = 0.x + 2 = 1**.
We'll call this line as **hypothesis** and denote the equation as **h(x)** that is **h(x) = x<sub>1</sub>.x + x<sub>0</sub>** and our case **h(x) = 2**.
We can trace our initial hypothesis as follows:-

![fig](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/Guide%20materials/linear%20regression%203.png)

We can make prediction using our hypothesis by inserting the value of **x** in **h(x)** and getting the predicted value or we can say value returned after solving the value of hypothesis at **x** equal to the value at which we want to predict.
Still our hypothesis is not perfect and cannot be used sucessfully for prediction because of the error present in it.

As you remember we have providet three points earlier to trace them, let's call that our **training set** and **m** number of training example in our case we have **3** examples. So our objective is minimize the error on that training set, but what is error.

We can denote error **c = (1/m).∑<sub>k=1</sub><sup>N</sup>(h(x<sup>k</sup>) - y<sup>k</sup>)<sup>2</sup>**

In training set lets say **(x<sup>k</sup>, y<sup>k</sup>)** is our **kth** data. The cost function is summation of error in our prediction which can be better illustrated in the following figure.

![fig](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/Guide%20materials/linear%20regression%204.png)

In the figure we can see that the difference between the actual training point and the predicted point is represented by the grey line.
So, our error function **c** is also called **cost function** and our objective as said earlied is to minimize our cost function by properly adjusting **x<sub>0</sub>** and **x<sub>1</sub>** that is **intercept** and **slope** of the line or our **hypothesis** and we can call **x<sub>0</sub>** and **x<sub>1<sub>** as our parameters which we will adjust so that the **cost function** minimizes.

There are many types of algorithm for doing that job of minimizing the cost function and we will use **Gradient Descent** for that job.
Let Us see how **Gradient Descent** works.

![fig](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/Guide%20materials/linear%20regression%205.png)

Here we'll do partial derivative of **cost function** and it is multiplied to **alpha** that is **learning rate** and is subtracted from the parameter and the value is assigned to the same parameter.
The above shown expression in only one step of gradient descent. In one step the point will move toward the local minima and as you take more steps. 

But how it will work practically.
Tempeorarily assume that in our case we have to adjust only **X<sub>1</sub>** and not **x<sub>0</sub>** means our **intercept** is adjusted but out slope is not. Then the graph of **c (cost)** vs **x<sub>1</sub>** will look like this roughly, which is not for scale. 

![gradient step](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/Guide%20materials/linear%20regression%206.png)

As you can see in above figure the gradient that is partial derivative of the **cost function** will point opposite to the minima but when we subtract it from the parameter it will point towards the minima.

The yellow arrow are the steps of the gradient descent. The length of the steps is proportional to the **alpha** that is learning rate.
It **learning rate** is too high then it will overtake the minima and the cost will start increasing inspite of decreasing but is **learning rate** is too low then gradient descent will take too many steps to converge to local minima thus increasing the computing.

As I told above that we have ignored the **x<sub>0</sub>**, but is we consider that paramater and plot a 3d graph where z axis is **cost**, x axis is **x<sub>1</sub>** and y axis is **x<sub>0</sub>**.

![two variable cost function graph](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/Guide%20materials/linear%20regression%207.png)

So in our example our line will become like this.

![anime ex](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/Guide%20materials/linear%20regression%208.gif)



