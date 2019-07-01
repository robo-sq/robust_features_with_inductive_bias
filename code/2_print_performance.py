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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.random.seed(247)
tf.set_random_seed(247)

#---------------------------------------------------------------------------------------------------------

all_models = ['LocalNet'] 
noise_status =   [False, True, False]
adv_status =     [False, False, True]

# save path
results_path = '../results'
model_path = utils.make_directory(results_path, 'model_params')
save_path = utils.make_directory(results_path, 'conv_filters')

# dataset path
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

with open(os.path.join(results_path, 'performance.tsv'), 'wb') as f:

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
            print('%s\t%.3f\t%.3f'%(name, roc, pr))
            f.write("%s\t%.3f\t%.3f\n"%(name, roc, pr))


                # # get 1st convolution layer filters
                # fmap = nntrainer.get_activations(sess, test, layer='conv1d_0_active')
                # W = visualize.activation_pwm(fmap, X=test['inputs'], threshold=0.5, window=19)

                # # plot 1st convolution layer filters
                # fig = visualize.plot_filter_logos(W, nt_width=50, height=100, norm_factor=None, num_rows=10)
                # fig.set_size_inches(100, 100)
                # outfile = os.path.join(save_path, name+'_conv_filters.pdf')
                # fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
                # plt.close()

                # # save filters as a meme file for Tomtom 
                # output_file = os.path.join(save_path, name+'.meme')
                # utils.meme_generate(W, output_file, factor=None)

                # # clip filters about motif to reduce false-positive Tomtom matches 
                # W = np.squeeze(np.transpose(W, [3, 2, 0, 1]))
                # W_clipped = helper.clip_filters(W, threshold=0.5, pad=3)
                
                # # since W is different format, have to use a different function
                # output_file = os.path.join(save_path, name+'_clip.meme')
                # helper.meme_generate(W_clipped, output_file, factor=None) 



