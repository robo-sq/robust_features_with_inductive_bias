from __future__ import print_function

import os, sys
import h5py
import numpy as np

import tensorflow as tf

sys.path.append('../..')
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency, metrics


def load_synthetic_dataset(filepath, verbose=True):
    # setup paths for file handling

    trainmat = h5py.File(filepath, 'r')

    if verbose:
        print("loading training data")
    X_train = np.array(trainmat['X_train']).astype(np.float32)
    y_train = np.array(trainmat['Y_train']).astype(np.float32)

    if verbose:
        print("loading cross-validation data")
    X_valid = np.array(trainmat['X_valid']).astype(np.float32)
    y_valid = np.array(trainmat['Y_valid']).astype(np.int32)

    if verbose:
        print("loading test data")
    X_test = np.array(trainmat['X_test']).astype(np.float32)
    y_test = np.array(trainmat['Y_test']).astype(np.int32)


    X_train = np.expand_dims(X_train, axis=3).transpose([0,2,3,1])
    X_valid = np.expand_dims(X_valid, axis=3).transpose([0,2,3,1])
    X_test = np.expand_dims(X_test, axis=3).transpose([0,2,3,1])

    train = {'inputs': X_train, 'targets': y_train}
    valid = {'inputs': X_valid, 'targets': y_valid}
    test = {'inputs': X_test, 'targets': y_test}

    return train, valid, test


def load_synthetic_models(filepath, dataset='test'):
    # setup paths for file handling

    trainmat = h5py.File(filepath, 'r')
    if dataset == 'train':
        return np.array(trainmat['model_train']).astype(np.float32)
    elif dataset == 'valid':
        return np.array(trainmat['model_valid']).astype(np.float32)
    elif dataset == 'test':
        return np.array(trainmat['model_test']).astype(np.float32)



def load_model(model_name, input_shape, outut_shape):

    # import model
    if model_name == 'cnn_4':
        from models import cnn_4 as genome_model
    elif model_name == 'cnn_4_noreg':
        from models import cnn_4_noreg as genome_model
    elif model_name == 'cnn_4_exp':
        from models import cnn_4_exp as genome_model
    elif model_name == 'cnn_25':
        from models import cnn_25 as genome_model
    elif model_name == 'cnn_25_noreg':
        from models import cnn_25 as genome_model
    elif model_name == 'cnn_25_exp':
        from models import cnn_25_exp as genome_model
    elif model_name == 'cnn_deep':
        from models import cnn_deep as genome_model
    elif model_name == 'cnn_deep_noreg':
        from models import cnn_deep_noreg as genome_model
    elif model_name == 'cnn_deep_exp':
        from models import cnn_deep_exp as genome_model
    elif model_name == 'mlp':
        from models import mlp as genome_model

    # load model specs
    model_layers, optimization = genome_model.model(input_shape,
                                                    output_shape)

    return model_layers, optimization, genome_model



def backprop(X, params, layer='output', class_index=None, batch_size=128, method='guided'):
    """wrapper for backprop/guided-backpro saliency"""

    tf.reset_default_graph()

    # build new graph
    model_layers, optimization, genome_model = load_model(params['model_name'], params['input_shape'], 
                                                   params['dropout_status'], params['l2_status'], params['bn_status'])

    nnmodel = nn.NeuralNet()
    nnmodel.build_layers(model_layers, optimization, method=method, use_scope=True)
    nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

    # setup session and restore optimal parameters
    sess = utils.initialize_session(nnmodel.placeholders)
    nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

    # backprop saliency
    if layer == 'output':
        layer = list(nnmodel.network.keys())[-2]

    saliency = nntrainer.get_saliency(sess, X, nnmodel.network[layer], class_index=class_index, batch_size=batch_size)

    sess.close()
    tf.reset_default_graph()
    return saliency



def smooth_backprop(X, params, layer='output', class_index=None, num_average=50):
    """wrapper for backprop/guided-backpro saliency"""

    tf.reset_default_graph()

    # build new graph
    model_layers, optimization, genome_model = load_model(params['model_name'], params['input_shape'], 
                                                   params['dropout_status'], params['l2_status'], params['bn_status'])

    nnmodel = nn.NeuralNet()
    nnmodel.build_layers(model_layers, optimization, method='backprop', use_scope=True)
    nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

    # setup session and restore optimal parameters
    sess = utils.initialize_session(nnmodel.placeholders)
    nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

    # backprop saliency
    if layer == 'output':
        layer = list(nnmodel.network.keys())[-2]


    saliency = np.zeros(X.shape)
    for i, x in enumerate(X):
        if np.mod(i,200) == 0:
            print('%d out of %d'%(i, len(X)))

        x = np.expand_dims(x, axis=0)
        shape = list(x.shape)
        shape[0] = num_average
        
        noisy_saliency = nntrainer.get_saliency(sess, x+np.random.normal(scale=0.1, size=shape), nnmodel.network[layer], class_index=class_index, batch_size=num_average)
        saliency[i,:,:,:] = np.mean(noisy_saliency, axis=0)

    return saliency




def clip_filters(W, threshold=0.5, pad=3):
    num_filters, _, filter_length = W.shape

    W_clipped = []
    for w in W:
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=0)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, filter_length)
            W_clipped.append(w[:,start:end])
        else:
            W_clipped.append(w)

    return W_clipped



def meme_generate(W, output_file='meme.txt', prefix='filter', factor=None):

    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j in range(len(W)):
        if factor:
            pwm = utils.normalize_pwm(W[j], factor=factor)
        else:
            pwm = W[j]
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (pwm.shape[1], pwm.shape[1]))
        for i in range(pwm.shape[1]):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[:,i]))
        f.write('\n')

    f.close()


def match_hits_to_ground_truth(file_path, motifs, size=30):
    
    # get dataframe for tomtom results
    df = pd.read_csv(file_path, delimiter='\t')
    
    # loop through filters
    best_qvalues = np.ones(size)
    best_match = np.zeros(size)
    for name in np.unique(df['#Query ID'].as_matrix()):
        filter_index = int(name.split('r')[1])

        # get tomtom hits for filter
        subdf = df.loc[df['#Query ID'] == name]
        targets = subdf['Target ID'].as_matrix()

        # loop through ground truth motifs
        for k, motif in enumerate(motifs): 

            # loop through variations of ground truth motif
            for motifid in motif: 

                # check if there is a match
                index = np.where((targets == motifid) ==  True)[0]
                if len(index) > 0:
                    qvalue = subdf['q-value'].as_matrix()[index]

                    # check to see if better motif hit, if so, update
                    if best_qvalues[filter_index] > qvalue:
                        best_qvalues[filter_index] = qvalue
                        best_match[filter_index] = k 

    # get the minimum q-value for each motif
    min_qvalue = np.zeros(13)
    for i in range(13):
        index = np.where(best_match == i)[0]
        if len(index) > 0:
            min_qvalue[i] = np.min(best_qvalues[index])

    match_index = np.where(best_qvalues != 1)[0]
    match_fraction = len(match_index)/float(size)

    return best_qvalues, best_match, min_qvalue, match_fraction 
