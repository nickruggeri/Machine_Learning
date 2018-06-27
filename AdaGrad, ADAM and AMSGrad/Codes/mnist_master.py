from my_mini_batch import *
from adam import *
from adagrad import *
from amsgrad import *
import matplotlib.pyplot as plt
import numpy as np

from utils import mnist_dataset
#generate the data
X, X_test, y, y_test= mnist_dataset()

# tranform the labels: we want to make binary classification:
# a number is labelled with 1 if it is a 7, with 0 otherwise
y = (y==7)

# to plot the learning curves, we average N learning curves for every method,
# in order to reduce the variance of the curve plotted
N = 2     
maxit = 2000  
lambd = 0.1      # regualarization coefficient

losses_adam_avg = np.zeros(maxit+1)
losses_adagrad_avg = np.zeros(maxit+1)
losses_sgd_avg = np.zeros(maxit+1)
losses_ams_avg = np.zeros(maxit+1)

for i in range(N):
	print('{}-th round of training'.format(i+1))

	theta_adam,losses_adam = adam(X,y,cost = cross_entropy,link = sigmoid, batch_size = 170,
	                     alpha = 0.01, lambd = lambd, maxit = maxit)

	theta_adagrad,losses_adagrad = adagrad(X,y,cost = cross_entropy,link = sigmoid, batch_size = 170,
	                     alpha = 0.01, lambd = lambd, maxit = maxit)

	theta_sgd,losses_sgd = sgd(X,y,cost = cross_entropy,link = sigmoid, batch_size = 170,
	                     alpha = 0.01, lambd = lambd, maxit = maxit)

	theta_ams,losses_ams = amsgrad(X,y,cost = cross_entropy,link = sigmoid, batch_size = 170,
	                     alpha = 0.01, lambd = lambd, maxit = maxit)


	losses_sgd_avg += losses_sgd
	losses_adagrad_avg += losses_adagrad
	losses_adam_avg += losses_adam
	losses_ams_avg += losses_ams

losses_adam_avg /= N
losses_adagrad_avg /= N
losses_sgd_avg /= N
losses_ams_avg /= N



# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, len(losses_sgd_avg)), losses_sgd_avg)
plt.plot(np.arange(0, len(losses_adam_avg)), losses_adam_avg)
plt.plot(np.arange(0, len(losses_adagrad_avg)), losses_adagrad_avg)
plt.plot(np.arange(0, len(losses_ams_avg)), losses_ams_avg)
fig.suptitle("Training Loss, averaged over {} trainings".format(N))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(['Mini Batch Sgd', 'ADAM', 'AdaGrad', 'AMSGrad'])

# save figure to jpg
fig.savefig('Error convergence.jpg')
plt.show()