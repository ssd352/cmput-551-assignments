from __future__ import division  # floating point division
import numpy as np
import utilities as utils
from scipy.stats import norm

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
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

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 0

    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set self.numclasses and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """

        ### YOUR CODE HERE
        self.numfeatures = Xtrain.shape[1] - (1 if not self.params['usecolumnones'] else 0)
        self.numclasses = len(set(ytrain))
        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.ones(origin_shape)
        self.stds = np.zeros(origin_shape)
        self.priors = np.fromiter((np.count_nonzero(ytrain == label) for label in range(self.numclasses)), dtype=np.float) / ytrain.shape[0]
        
        ### YOUR CODE HERE
        # for feature in self.numfeatures:
        for label in range(self.numclasses):
            filter = Xtrain[np.where(ytrain == label)[0], :self.numfeatures]
            self.means[label, :] = np.mean(filter, axis=0)
            self.stds[label, :] = np.std(filter, axis=0)
            # print(self.stds)
        ### END YOUR CODE

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE

        for cnt in range(Xtest.shape[0]):
            sample = Xtest[cnt, :]
            max_prod = 0
            max_ind = -1
            for label in range(self.numclasses):
                _ = (norm.pdf(sample[feature], loc=self.means[label][feature], scale=self.stds[label][feature]) for feature in range(Xtest.shape[1] - 1))
                likelihoods = np.fromiter(_, dtype=np.float)
                pr = self.priors[label] * np.prod(likelihoods)
                if max_prod < pr:
                    max_prod = pr
                    max_ind = label

            ytest[cnt] = max_ind 
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0
        num_samples = X.shape[0]

        ### YOUR CODE HERE
        for cnt in range(num_samples):
            cost += -y[cnt] * np.log(utils.sigmoid(np.dot(theta, X[cnt, :])))
            cost += -(1 - y[cnt]) * np.log(1 - utils.sigmoid(np.dot(theta, X[cnt, :])))
        ### END YOUR CODE

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE
        grad = X.T @ (utils.sigmoid(X @ theta) - y)
        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        self.weights = np.zeros(Xtrain.shape[1],)
        
        etha = 0.0001
        threshold = 0.0001
        delta = 1000
        ### YOUR CODE HERE
        while delta > threshold:
            grad = self.logit_cost_grad(self.weights, Xtrain, ytrain)
            # print(grad.shape)
            delta = np.amax(np.abs(etha * grad))
            self.weights += - etha * grad
            # print(delta)
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        # for cnt in range(ytest.shape[0]):
        #     ytest[cnt] = round()
        ytest = np.round(utils.sigmoid(Xtest @ self.weights))
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 100}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        # a_hidden = self.transfer(np.dot(self.w_input, inputs))
        a_hidden = self.transfer(inputs @ self.w_input)

        # output activations
        # a_output = self.transfer(np.dot(self.w_output, a_hidden))
        a_output = self.transfer(a_hidden @ self.w_output)

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """


        ### YOUR CODE HERE
        # a_hidden = self.transfer(np.dot(self.w_input, x))
        # a_output = self.transfer(np.dot(self.w_output, a_hidden))
        a_hidden, a_output = self.feedforward(x)
        # loss = - y * np.log(a_output) - (1 - y) * np.log(1 - a_output)
        dloss = (- y / a_output + (1 - y) / (1 - a_output))[0]

        # print(self.dtransfer(a_hidden @ self.w_output))
        nabla_output = np.ones(self.w_output.shape)
        nabla_output[:, 0] = dloss * self.dtransfer(a_hidden @ self.w_output) * a_hidden
        # np.expand_dims(nabla_output, axis=1)
        # print(nabla_output.shape, self.w_output.shape)
        
        nabla_input = np.ones(self.w_input.shape)
        nabla_input *= dloss * self.dtransfer(a_hidden @ self.w_output)
        for i in range(nabla_input.shape[0]):
            for j in range(nabla_input.shape[1]):
                nabla_input[i, j] *= self.w_output[j, 0] * self.dtransfer(x @ self.w_input[:, j]) * x[i] 

        
        # print(self.w_output.shape, self.dtransfer(x @ self.w_input), x.shape)
        # nabla_input = dloss * self.dtransfer(a_hidden @ self.w_output) * x @ self.dtransfer(x @ self.w_input) @ self.w_output 
        ### END YOUR CODE

        # print(nabla_input.shape, self.w_input.shape)
        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    # TODO: implement learn and predict functions

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        # self.weights = np.zeros(Xtrain.shape[1],)
        num_features = Xtrain.shape[1]
        num_samples = Xtrain.shape[0]
        nh = self.params["nh"]
        self.w_input = np.random.rand(num_features, nh)
        self.w_output = np.random.rand(nh, 1)
        
        
        ### YOUR CODE HERE
        step_size = self.params["stepsize"]
        for _ in range(self.params['epochs']):
            for cnt in range(num_samples):
                nabla_input, nablab_output = self.backprop(Xtrain[cnt, :], ytrain[cnt])
                self.w_input += -step_size * nabla_input
                self.w_output +=  -step_size * nablab_output
            # grad = self.logit_cost_grad(self.weights, Xtrain, ytrain)
            # print(grad.shape)
            # delta = np.amax(np.abs(etha * grad))
            # self.weights += - etha * grad
            # print(delta)
            
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        # for cnt in range(ytest.shape[0]):
        #     ytest[cnt] = round()
        _, output = self.feedforward(Xtest)
        ytest = np.round(output)
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': None, 'kernel': None, 'stepsize': 1e-6, 'tolerance': None}
        self.reset(parameters)
        self.kernel = None
        # if self.params['kernel'] == "None":
        #    self.kernel = self.linear
        if self.params['kernel'] == "linear":
            self.kernel = self.linear
        elif self.params['kernel'] == "hamming":
            self.kernel = self.hamming

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        Ktrain = None

        ### YOUR CODE HERE
        
        
        self.num_samples = Xtrain.shape[0]
        self.Xtrain = Xtrain
        if self.params['kernel'] is None:
            Ktrain = Xtrain
        # elif self.params['kernel'] == "linear":
        #     Ktrain = np.dot(Xtrain, Xtrain)
        else:
            Ktrain = np.zeros(shape=(Xtrain.shape[0], self.num_samples))
            for i in range(Xtrain.shape[0]):
                for j in range(self.num_samples):
                    Ktrain[i, j] = self.kernel(Xtrain[i, :], Xtrain[j, :])
                    # print(Ktrain[i, j])
        ### END YOUR CODE

        self.weights = np.zeros(Ktrain.shape[1],)

        # etha = 1e-5
        etha = self.params["stepsize"]
        # threshold = 1e-2
        if self.params['tolerance'] is None:
            threshold = etha * 10
        else:
            threshold = self.params['tolerance']
        delta = 1000
        ### YOUR CODE HERE
        cost = self.logit_cost(self.weights, Ktrain, ytrain)
        # print("cost: {}".format(cost))
        while delta > threshold:
            grad = self.logit_cost_grad(self.weights, Ktrain, ytrain)
            # print(grad.shape)
            delta = np.amax(np.abs(etha * grad))
            self.weights += - etha * grad
            cost = self.logit_cost(self.weights, Ktrain, ytrain)
            # print("cost: {}\ndelta: {}".format(cost, delta))

        ### END YOUR CODE

        self.transformed = Ktrain # Don't delete this line. It's for evaluation.

    # TODO: implement necessary functions

    # def logit_cost_grad(self, theta, X, y):
    #     """
    #     Compute gradients of the cost with respect to theta.
    #     """

    #     grad = np.zeros(len(theta))

    #     ### YOUR CODE HERE
    #     grad = X.T @ (utils.sigmoid(X @ theta) - y)
    #     ### END YOUR CODE

        # return grad

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        # for cnt in range(ytest.shape[0]):
        #     ytest[cnt] = round()
        if self.params["kernel"] == "None":
            Ktest = Xtest
        else:
            Ktest = np.zeros(shape=(Xtest.shape[0], self.num_samples))
            for i in range(Ktest.shape[0]):
                for j in range(Ktest.shape[1]):
                    Ktest[i, j] = self.kernel(Xtest[i, :], self.Xtrain[j, :])
        ytest = np.round(utils.sigmoid(Ktest @ self.weights))
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

    def hamming(self, x1, x2):
        # return np.count_nonzero(x1 == x2)
        # print(x1, x2)
        return np.count_nonzero(np.abs(x1 - x2) < 1e-2)
    
    def linear(self, x1, x2):
        return np.dot(x1, x2) / np.linalg.norm(x1) / np.linalg.norm(x2)


# ======================================================================

def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"

    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)

    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(X[0, :], y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()

if __name__ == "__main__":
    main()
