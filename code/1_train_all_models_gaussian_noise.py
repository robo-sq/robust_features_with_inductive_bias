from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import helper
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#------------------------------------------------------------------------------------------------


all_models = ['cnn_25_noreg', 'cnn_deep_noreg'] 

# save path
results_path = '/content/drive/My Drive/results'

model_path = utils.make_directory(results_path, 'model_params')


# dataset path
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

# loop through models
for model_name in all_models:
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
    num_epochs = 10
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
        noisy_train = {'inputs': train['inputs'] + np.random.normal(scale=0.1, size=train['inputs'].shape),
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
        # save model
        nntrainer.save_model(sess)

        # early stopping
        if not nntrainer.early_stopping(loss, patience):
            break