import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))


def toy_dataset():
    (X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
    	cluster_std=1.05, random_state=20)
    X = np.c_[np.ones((X.shape[0])), X]
    return X,y


# Import MINST data
def mnist_dataset():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
    y_train = mnist.train.labels
    X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
    y_test = mnist.test.labels
    return X_train, X_test, y_train, y_test

def initialize_weights(p):
    return np.random.uniform(size = p)

def make_predictions(X,W,link):
    return link(X.dot(W))

def cross_entropy(y,preds):
    return -np.sum(y*np.log(preds)+(1-y)*np.log(1-preds))/y.shape[0]

def compute_gradient(preds, X, y, w, cost = 'cross entropy' ,link = 'sigmoid', reg = 'l2', lambd = 0.1):
    if cost == 'cross entropy' and link == 'sigmoid' and reg =='l2': 
        return X.T.dot(preds-y)/y.shape[0] + lambd*w
