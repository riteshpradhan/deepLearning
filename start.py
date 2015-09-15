import cPickle, gzip, numpy
import theano
from theano import tensor as T

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

#Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def know_mnist_data():
    print "jpt shape: ", numpy.asarray([[[2,3], [23,23], [33,33]],[[2,3], [23,23], \
        [33,33]],[[2,3], [23,23], [33,33]],[[2,3], [23,23], [33,33]]]).shape        #4 * 3 * 2
    print len(train_set)
    print train_set[0].shape
    print "each data shape ", train_set[0][0].shape
    print "single image: ", train_set[0][0]

know_mnist_data()

# train_set_x, train_set_y = shared_dataset(train_set)
# valid_set_x, valid_set_y = shared_dataset(valid_set)
# test_set_x, test_set_y = shared_dataset(test_set)

# batch_size = train_set_x[2*batch_size : 3*batch_size]
# label = train_set_y[2*batch_size : 3*batch_size]

