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
from deepomics import utils, fit

#------------------------------------------------------------------------------------------------

num_trials = 5

all_models = ['cnn_4', 'cnn_4_noreg', 'cnn_4_exp',
              'cnn_25', 'cnn_25_noreg', 'cnn_25_exp',
              'cnn_deep', 'cnn_deep_noreg', 'cnn_deep_exp',
              'mlp'] 

# save path
results_path = '../results'
model_path = utils.make_directory(results_path, 'model_params')
metrics_path = utils.make_directory(results_path, 'train_metrics')

# dataset path
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

# loop through models
for trial in range(num_trials):
    for model_name in all_models:
        tf.reset_default_graph()

        name = model_name + '_noise_' + str(trial)
        print('model: ' + name)

        file_path = os.path.join(model_path, name)

        # load model parameters
        model_layers, optimization, _ = helper.load_model(model_name, 
                                                          input_shape,
                                                          output_shape)

        # build neural network class
        nnmodel = nn.NeuralNet()
        nnmodel.build_layers(model_layers, optimization, supervised=True)

        # create neural trainer class
        nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

        # initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # set data in dictionary
        num_epochs = 120
        batch_size = 100
        verbose = 2
        shuffle = True
        train_metric = []
        test_metric = []
        valid_metric = []
        for epoch in range(num_epochs):
            if verbose >= 1:
                sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))
            else:
                if epoch % 10 == 0:
                    sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1,num_epochs))

            # training set
            noisy_train = {'inputs': train['inputs'] + np.random.normal(scale=0.1, size=train['inputs'].shape),
                           'targets': train['targets']}
            train_loss = nntrainer.train_epoch(sess, noisy_train,
                                                batch_size=batch_size,
                                                verbose=verbose,
                                                shuffle=shuffle)

            # save validcation metrics
            loss, mean_vals, error_vals = nntrainer.test_model(sess, valid,
                                                                    name="valid",
                                                                    batch_size=1024,
                                                                    verbose=verbose)
            valid_metric.append([loss, mean_vals])

            # save train metrics
            loss, mean_vals, error_vals = nntrainer.test_model(sess, train,
                                                                    name="train",
                                                                    batch_size=1024,
                                                                    verbose=verbose)
            train_metric.append([loss, mean_vals])

            # save test metrics
            loss, mean_vals, error_vals = nntrainer.test_model(sess, test,
                                                                    name="test",
                                                                    batch_size=1024,
                                                                    verbose=verbose)
            test_metric.append([loss, mean_vals])

            # save early stopping model
            nntrainer.save_model(sess)

        # save model at last epoch
        nntrainer.save_model(sess, 'last')

        # save results
        with open(os.path.join(metrics_path, name+'_metrics.pickle'), 'wb') as f:
            cPickle.dump(train_metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(valid_metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(test_metric, f, protocol=cPickle.HIGHEST_PROTOCOL)

