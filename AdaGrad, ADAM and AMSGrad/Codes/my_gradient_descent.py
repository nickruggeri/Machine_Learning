import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

def dataset():
    (X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
    	cluster_std=1.05, random_state=20)
    X = np.c_[np.ones((X.shape[0])), X]
    return X,y

def initialize_weights(p):
    return np.random.uniform(size = p)

def make_predictions(X,W,link):
    return link(X.dot(W))

def cross_entropy(y,preds):
    return -np.sum(y*np.log(preds)+(1-y)*np.log(1-preds))/y.shape[0]

def compute_gradient(preds,X,y,cost=cross_entropy,link = sigmoid):
    if cost == cross_entropy and link == sigmoid:
        return X.T.dot(preds-y)/y.shape

def gradient_descent(X,y,cost = cross_entropy,link=sigmoid,
                     alpha = 0.01,eps = 0.0001, maxit = 1000):
    W = initialize_weights(X.shape[1])
    losses = []
    preds = make_predictions(X,W,link)
    losses.append(cost(y,preds))
    it = 0
    while True:
        it += 1
        print("Iteration n.{}".format(it))
        l_prev = losses[-1]
        gradient = compute_gradient(preds,X,y,cost,link)
        W -= alpha*gradient
        preds = make_predictions(X,W,link)
        l_new = cost(y,preds)
        losses.append(l_new)
        if l_prev - l_new < eps or it == maxit:
            break
    return W,losses

#generate the data
X,y = dataset()

# plot the points
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)

theta,losses = gradient_descent(X,y,cost = cross_entropy,link = sigmoid,
                     alpha = 0.01,eps = 0.0001, maxit = 1000)

Y = (-theta[0] - (theta[1] * X)) / theta[2]

plt.plot(X, Y, "r-")

# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, len(losses)), losses)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()