from __future__ import division  # floating point division
import numpy as np
import math

import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.pinv((Xless.T @ Xless)/numsamples), Xless.T),ytrain)/numsamples
        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.5}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain
        self.lmd = self.params["regwgt"]
        self.weights = np.dot(np.dot(np.linalg.pinv((Xless.T @ Xless) / numsamples + self.lmd * np.eye(Xtrain.shape[1])), Xless.T),ytrain)/numsamples
        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest

class LassoLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'tol': 1e-4, "max_iter": 100000, 'regwgt': 0.5}
        # self.weights = np.random.randn()
        self.reset(parameters)

    def proximal(self, w, threshold):
        ind1 = np.where(w > threshold)
        w[ind1] -= threshold
        ind2 = np.where(w < -threshold)
        w[ind2] += threshold
        return w


    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        dim = Xtrain.shape[1]
        self.weights = np.zeros(dim)
        self.tolerance = self.params["tol"]
        err = np.Infinity
        numsamples = Xtrain.shape[0]
        xx = Xtrain.T @ Xtrain / numsamples
        xy = Xtrain.T @ ytrain / numsamples
        self.etha = 0.5 / np.linalg.norm(xx, ord="fro")
        self.max_iter = self.params["max_iter"]
        # Xless = Xtrain
        self.lmd = self.params["regwgt"]

        cw = np.linalg.norm(Xtrain @ self.weights - ytrain) ** 2 / 2 / numsamples + self.lmd * np.linalg.norm(self.weights, ord=1) 
        cnt = 0
        while np.abs(cw - err) >= self.tolerance and cnt < self.max_iter:
            err = cw
            cnt += 1
            self.weights = self.proximal(self.weights - self.etha * xx @ self.weights + self.etha * xy, self.etha * self.lmd)
            cw = np.linalg.norm(Xtrain @ self.weights - ytrain) ** 2 / 2 / numsamples + self.lmd * np.linalg.norm(self.weights, ord=1)
        # self.weights = np.dot(np.dot(np.linalg.pinv((Xless.T @ Xless) / numsamples + self.lmd * np.eye(Xtrain.shape[1])), Xless.T),ytrain)/numsamples
        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest    

class BGDLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'tol': 1e-4, "max_iter": 100000, }
        # self.weights = np.random.randn()
        self.reset(parameters)
    
    
        

    def line_search(self, wt, cost_func, g):
        etha_max = 1
        tau = 0.7
        tolerance = self.tolerance
        etha = etha_max
        w = np.copy(wt)
        obj = cost_func(w)
        max_iter = self.max_iter
        for _ in range(max_iter):
            w = wt - etha * g
            if cost_func(w) < obj - tolerance:
                return w, etha
            etha *= tau
        return wt, 0

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        dim = Xtrain.shape[1]
        self.weights = np.random.randn(dim)
        self.tolerance = self.params["tol"]
        err = np.Infinity
        numsamples = Xtrain.shape[0]

        self.max_iter = self.params["max_iter"]
        # Xless = Xtrain
        def cost(w):
            return np.linalg.norm(Xtrain @ w - ytrain) ** 2 / 2 / numsamples

        # cw = np.linalg.norm(Xtrain @ self.weights - ytrain) ** 2 / 2 / numsamples
        cw = cost(self.weights)
        cnt = 0
        while np.abs(cw - err) >= self.tolerance and cnt < self.max_iter:
            err = cost(self.weights)
            cnt += 1
            g = Xtrain.T @ (Xtrain @ self.weights - ytrain) / numsamples
            etha = self.line_search(self.weights, cost, g)[1]
            self.weights -= etha * g
            cw = cost(self.weights)
            # cw = np.linalg.norm(Xtrain @ self.weights - ytrain) ** 2 / 2 / numsamples
        # self.weights = np.dot(np.dot(np.linalg.pinv((Xless.T @ Xless) / numsamples + self.lmd * np.eye(Xtrain.shape[1])), Xless.T),ytrain)/numsamples
        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest  

class SGDLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {"num_epochs": 1000}
        # self.weights = np.random.randn()
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        dim = Xtrain.shape[1]
        self.weights = np.random.randn(dim)
        self.num_epochs = self.params["num_epochs"]
        numsamples = Xtrain.shape[0]
        etha_0 = .01
        # err = np.Infinity
        for i in range(self.num_epochs):
            for j in np.random.permutation(numsamples):
                xj = Xtrain[j, :]
                # print(xj.shape, ytrain.shape)
                g = (xj.T @ self.weights - ytrain[j]) * xj
                # print(g)
                etha = etha_0 / (i + 1)
                self.weights -= etha * g
            # Xless = Xtrain
            # self.lmd = self.params["regwgt"]

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest    
