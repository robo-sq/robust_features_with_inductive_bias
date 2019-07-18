from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
import helper
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency, metrics
from deepomics import utils, fit
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#------------------------------------------------------------------------------------------------

num_trials = 5

# save path
results_path = '../results'
results_path = '/content/drive/My Drive/results'

isMnist = False

if isMnist:

  all_models = ['cnn_deep_noreg_mnist'] 
  model_path = utils.make_directory(results_path, 'model_params/mnist')
  metrics_path = utils.make_directory(results_path, 'train_metrics')

  # Import MINST data
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  train = {}
  valid = {}
  test = {}


  train['inputs'] = np.reshape(mnist.train.images, [-1, 28, 28, 1])
  train['targets'] = mnist.train.labels

  valid['inputs'] = np.reshape(mnist.validation.images, [-1, 28, 28, 1])
  valid['targets'] = mnist.validation.labels

  test['inputs'] = np.reshape(mnist.test.images, [-1, 28, 28, 1])
  test['targets'] = mnist.test.labels

else:
  all_models = ['cnn_deep_noreg'] 
  model_path = utils.make_directory(results_path, 'model_params')

  # dataset path
  data_path = '../data/Synthetic_dataset.h5'
  train, valid, test = helper.load_synthetic_dataset(data_path)

eps_list = np.linspace(0,1,51)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

res_dict = {}

# loop through models
for model_name in all_models:
    for eps in eps_list:
        res_dict[eps] = []
        for idx in range(num_trials):

            tf.reset_default_graph()

            name = model_name+'_noise'
            print('model: ' + name)

            file_path = os.path.join(model_path, name)

            # load model parameters
            model_layers, optimization, _ = helper.load_model(model_name, 
                                                              input_shape,
                                                              output_shape)

            # build neural network class
            nnmodel = nn.NeuralNet()
            nnmodel.build_layers(model_layers, optimization, supervised=True)

            nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

            # initialize session
            sess = utils.initialize_session()

            # set data in dictionary
            data = {'train': train, 'valid': valid, 'test': test}

            # set data in dictionary
            num_epochs = 20
            batch_size = 100
            patience = 25
            verbose = 2
            shuffle = True
            for epoch in range(num_epochs):
                if verbose >= 1:
                    sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))
                else:
                    if epoch % 10 == 0:
                        sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1,num_epochs))

                # training set
                noisy_train = {'inputs': train['inputs'] + np.random.uniform(low=-eps, high=eps, size=train['inputs'].shape),
                               'targets': train['targets']}
                train_loss = nntrainer.train_epoch(sess, noisy_train,
                                                    batch_size=batch_size,
                                                    verbose=verbose,
                                                    shuffle=shuffle)

                # save cross-validcation metrics
                loss, mean_vals, error_vals = nntrainer.test_model(sess, valid,
                                                                        name="valid",
                                                                        batch_size=batch_size,
                                                                        verbose=verbose)

            # get performance metrics
            predictions = nntrainer.get_activations(sess, train, 'output')
            acc = metrics.accuracy(train['targets'], predictions)
            print('Epsilon: ' + str(eps))
            print('Trial: ' + str(idx+1))
            print(acc[0])
            res_dict[eps].append(acc)
        print('Mean for eps=' + str(eps))
        print(np.mean(res_dict[eps]))

        with open(os.path.join(results_path, model_name+'_acc.pickle'), 'wb') as f:
                    cPickle.dump(res_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)

        print(res_dict)
