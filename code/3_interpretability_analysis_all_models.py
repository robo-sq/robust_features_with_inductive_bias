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
import helper
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.random.seed(247)
tf.set_random_seed(247)

#---------------------------------------------------------------------------------------------------------


all_models = ['LocalNet'] 
noise_status =   [False, True, False]
adv_status =     [False, False, True]



methods = ['backprop', 'mutagenesis', 'smoothgrad']

num_trials = 5

# save path
results_path = '../results'
model_path = utils.make_directory(results_path, 'model_params')

# dataset path
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

test_model = helper.load_synthetic_models(data_path, dataset='test')

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

true_index = np.where(test['targets'][:,0] == 1)[0]
X = test['inputs'][true_index]
X_model = test_model[true_index]

for method in methods:
    print(method)

    attribution_results ={}
    for model_name in all_models:
        
        for n, noise in enumerate(noise_status):

            for trial in range(num_trials):
                name = model_name
                if noise:
                    name += '_noise'
                if adv_status[n]:
                    name += '_adv'
                name = name + '_' + str(trial)

                print('model: ' + name)
                
                file_path = os.path.join(model_path, name)

                # attribution parameters for trained model
                params = {'model_name': model_name, 
                          'input_shape': input_shape, 
                          'output_shape': output_shape,
                          'model_path': file_path+'_last.ckpt',
                         }

                # get attribution scores
                if method == 'backprop':
                    X_attrib = helper.backprop(X, params, layer='output', class_index=None, method='backprop')
                elif method == 'smoothgrad':
                    X_attrib = helper.smooth_backprop(X, params, layer='output', class_index=None, num_average=50)
                elif method == 'mutagenesis':
                    X_attrib = helper.mutagenesis(X, params, layer='output', class_index=None)
  
                # quantify attribution scores with ground truth
                roc_score, pr_score = helper.interpretability_performance(X, X_attrib, X_model, method='l2norm')

                attribution_results[name] = [roc_score, pr_score]

        with open(os.path.join(results_path, method+'_last_scores.pickle'), 'wb') as f:
            cPickle.dump(attribution_results, f, protocol=cPickle.HIGHEST_PROTOCOL)


    # print results
    for key in backprop_results.keys():
        roc_score, pr_score = backprop_results[key]
        print('%s\t%.3f+/-%.3f\t%.3f+/-%.3f'%(key, 
                                              np.mean(roc_score), 
                                              np.std(roc_score),
                                              np.mean(pr_score), 
                                              np.std(pr_score)))
