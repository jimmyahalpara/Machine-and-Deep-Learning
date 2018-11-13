# **Tensorflow-Projects**
Here I'll be posting my tensorflow codes as I start Learning them
<h1>1.  My first program of linear regression</h1> 

My first program of linear regression was just plotting the hypothesis on the datasets randomly created using **numpy** and plotted using **matplotlib** in **python** and model trainded by **tensorflow** using **gradient descent algorithem**.

![plotting of hypothesis](https://github.com/jimmyahalpara/Tensorflow-Projects/blob/master/simple%20linear%20regression%20with%20tensorflow%20by%20me/Capture.PNG)

<h2> How linear regression and gradient descent works </h2>

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

