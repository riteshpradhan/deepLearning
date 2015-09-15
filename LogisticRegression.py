#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-09-05 09:50:19
# @Last Modified by:   ritesh
# @Last Modified time: 2015-09-08 15:06:29

import os
import sys
import timeit
import cPickle
import gzip

import theano
from theano import tensor as T
import numpy

class LogisticRegression:
    """ Logistic Regression for multiple class Classification

    W: Weight matrix
    b: bias vector
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters

        :type input: theano.tensor.TensorType
        :param input: input of one minibatch

        :type n_in: int
        :param n_in: number of input units, datapoints

        :type n_out: int
        :param n_out: number of input units, labels
        """
        self.W = theano.shared( value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                                                name="W",
                                                borrow=True
                                                )

        self.b = theano.shared( value=numpy.zeros((n_out, ), dtype=theano.config.floatX),
                                            name="b",
                                            borrow=True
                                            )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        """ Returns the mean of negative log-likelihood of prediction

        :type y: theano.tensor
        :param y: vector giving correct label to each examples
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ Returns a float representing the number of errors
        """

        if y.ndim != self.y_pred.ndim:
            raise TypeError('y must be same size of y_pred', 'y', y.type, 'y_pred', y_pred.type)
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError


def load_data(dataset):
    """ Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """


    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """ Loads dataset in shared variables
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


#optimization train
def sgd_optimization(dataset, learning_rate, n_epochs, batch_size):
    """
    Stochastic gradient descent optimization with minibatches

    :type dataset: string
    :param dataset: MNIST dataset path

    :type learning_rate: float
    :param learning_rate: learning rate used

    :type n_epochs: int
    :param n_epochs: max epochs to run

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    #build the model
    print "... building the model"

    index = T.lscalar()
    x = T.matrix('x')       #data for the rasterized images
    y = T.ivector('y')      # labels (int)

    # logistic regression Class
    classifierLR = LogisticRegression(input=x, n_in=28*28, n_out=10)
    cost = classifierLR.negative_log_likelihood(y)

    # test model (no updates)
    test_model = theano.function(
        inputs=[index],
        outputs=classifierLR.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    #validate model (no updates)
    validate_model = theano.function(
        inputs=[index],
        outputs=classifierLR.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    #compute the gradient of cost wrt W, b
    g_W = T.grad(cost=cost, wrt=classifierLR.W)
    g_b = T.grad(cost=cost, wrt=classifierLR.b)

    #updating expression
    updates = [(classifierLR.W, classifierLR.W - learning_rate * g_W),
                (classifierLR.b, classifierLR.b - learning_rate * g_b)]

    # Train model (theano function); updates
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]

        }
    )

    # Training model (early stopping with validation examples)
    print "... training the model"
    patience = 5000
    patience_inc = 2    # wait this much
    improved_threshold = 0.995  # relative improvement (significant)
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute loss on validation set
                validation_losses  = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    "Epoch: %i, minibatch: %i/%i, validation_error: %f %%" %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    #improve patience if good improvement
                    if this_validation_loss < best_validation_loss * improved_threshold:
                        patience = max(patience, iter * patience_inc)

                    best_validation_loss = this_validation_loss

                    #testing on test_set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            "Epoch : %i, minibatch %i/%i,"
                            " test error of best model %f %%"
                        ) % (
                            epoch,
                            minibatch_index,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    #save the best model
                    print "New best model found; saving ..."
                    with open('best_model.pkl', "w") as f:
                        cPickle.dump(classifierLR, f)

            if patience <= iter:
                done_looping = True
                break


    end_time = timeit.default_timer()
    print(
        (
            "Optimization Complete: best validation score : %f %%,"
            " test performance : %f %%"
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print "The code run for %d epochs, with %f epochs/sec" %(epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ("The code for file " + os.path.split(__file__)[1] + " ran for %.1fs" % ((end_time - start_time)))


#predict
def predict():
    """Example to load the trained model and predict labels
    """
    bestClassifierLR = cPickle.load(open("best_model.pkl"))

    predict_model = theano.function(
        inputs =[bestClassifierLR.input],
        outputs=bestClassifierLR.y_pred)

    dataset = "mnist.pkl.gz"
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in the test set : ")
    print predicted_values


#run
def main():
    dataset = 'mnist.pkl.gz'
    learning_rate = 0.1
    n_epochs = 1000
    batch_size = 600
    sgd_optimization(dataset, learning_rate, n_epochs, batch_size)

if __name__ == '__main__':
    # main()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   main()
    predict()