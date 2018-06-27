import numpy as np
import math

from utils import *

def amsgrad(X,y,cost = cross_entropy,link=sigmoid, reg='l2', batch_size = 1,
                     alpha = 0.01, lambd = 0.0001, maxit = 1000):
    beta_1 = 0.9
    beta_2 = 0.999
    m = 0
    v = 0
    v_hat = 0
    err = 10**(-8)
    W = initialize_weights(X.shape[1])
    n = X.shape[0]
    # np.random.RandomState(50)
    ind = np.random.permutation(np.arange(n))
    X = X[ind]
    y = y[ind]
    i = 0
    losses = []
    i_end = i+batch_size
    preds = make_predictions(X[i:i_end,:],W,link)
    losses.append(cost(y[i:i+batch_size],preds))
    it = 0
    while True:
        it += 1
        print("AMSGrad: iteration n.{}".format(it))
        gradient = compute_gradient(preds,X[i:i_end,:],y[i:i_end], W, lambd = lambd)
        m = beta_1 * m + (1-beta_1) * gradient
        v = beta_2 * v + (1-beta_2) * gradient**2
        v_hat = np.maximum(v, v_hat)
        W -= alpha*m/(np.sqrt(v_hat)+err)
        i = i_end
        i_end = i + batch_size
        if i_end > n:
            i_end = n
        if i == n:
            ind = np.random.permutation(np.arange(n))
            X = X[ind]
            y = y[ind]
            i = 0
            i_end = i + batch_size
        preds = make_predictions(X[i:i_end,:],W,link)
        l_new = cost(y[i:i_end],preds)
        if reg=='l2':
            l_new += lambd/2*np.linalg.norm(W, 2)**2
        losses.append(l_new)
        if it == maxit:
            break
    return W,losses