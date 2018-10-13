from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numruns = 5

    regressionalgs = {'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                'RidgeLinearRegression1': algs.RidgeLinearRegression({'regwgt': 0.0}),
                'RidgeLinearRegression2': algs.RidgeLinearRegression({'regwgt': 0.01}),
                'RidgeLinearRegression3': algs.RidgeLinearRegression({'regwgt': 1.0}),
                'BGD1': algs.BGDLinearRegression(),
                'Lasso1': algs.LassoLinearRegression(),
                'SGD1': algs.SGDLinearRegression(),
             }
    # for numFeatures in range(15, 385, 50):
    #     regressionalgs['FSLinearRegressionAsc{}'.format(numFeatures)] = algs.FSLinearRegression({'features': range(numFeatures) })
    #     regressionalgs['FSLinearRegressionRand{}'.format(numFeatures)] = algs.FSLinearRegression({'features': np.random.choice(range(385), size=numFeatures, replace=False)})
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
        {'regwgt': 0.0},
        {'regwgt': 0.01},
        {'regwgt': 1.0},
                      )
    numparams = len(parameters)
    
    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        # print(trainset[0].shape[1])
        

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(1):
            # params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                # learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error

    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p
        stderror = np.sqrt(np.sum((errors[learnername][:,:] - besterror) ** 2) / numruns)
        # print((errors[learnername][:,:] - besterror) ** 2)

        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))
        print('Standard error for {}: {}'.format(learnername, stderror))
