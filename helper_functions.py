import numpy as np
from scipy.optimize import fmin_tnc

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def fit(x, y):
    L = 0.0001  # The learning Rate
    epochs = 2000000  # The number of iterations to perform gradient descent
    numsamples = x.shape[0]
    theta = np.zeros((x.shape[1],1))
    for i in range(epochs):
        djw = (1 / numsamples) * np.dot(x.T, predict_prob(theta,x) - y)
        theta = theta - L * djw
    return theta

def predict_prob(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def predict_classes(theta,x):
    return (predict_prob(theta, x) >= 0.5).astype(int)
