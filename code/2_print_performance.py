from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt

import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency, metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
import helper
np.random.seed(247)
tf.set_random_seed(247)

#---------------------------------------------------------------------------------------------------------

all_models = ['cnn_4', 'cnn_4_noreg', 'cnn_4_exp',
              'cnn_25', 'cnn_25_noreg', 'cnn_25_exp',
              'cnn_deep', 'cnn_deep_noreg', 'cnn_deep_exp',
              'mlp'] 
noise_status =   [False, True, False]
adv_status =     [False, False, True]

# save path
results_path = '../results'
params_path = utils.make_directory(results_path, 'model_params')

# dataset path
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

with open(os.path.join(results_path, 'performance.tsv'), 'wb') as f:

    for i in range(len(dropout_status)):

        for n, noise in enumerate(noise_status):
            # loop through models
            for model_name in all_models:
                tf.reset_default_graph()

                # compile neural trainer
                name = model_name
                if noise:
                    name += '_noise'
                if adv_status[n]:
                    name += '_adv'
                print('model: ' + name)

                model_path = utils.make_directory(params_path, model_name)
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

                # set the best parameters
                nntrainer.set_best_parameters(sess)#, file_path=file_path+'_last.ckpt')

                # get performance metrics
                predictions = nntrainer.get_activations(sess, test, 'output')
                roc, roc_curves = metrics.roc(test['targets'], predictions)
                pr, pr_curves = metrics.pr(test['targets'], predictions)

                # print performance results
                f.write("%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\n"%(model_name, str(noise), str(dropout_status[i]), str(l2_status[i]), str(bn_status[i]), roc, pr))
