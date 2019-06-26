
def model(input_shape, dropout=True, l2=True, batch_norm=True):

    layer1 = {  'layer': 'input',
                'input_shape': input_shape
             }
    layer2 = {  'layer': 'conv1d',
                'num_filters': 24,
                'filter_size': 7,
                'padding': 'SAME',
                'norm': 'batch',
                'activation': 'exp',
                'dropout': 0.1,
                }
    layer3 = {  'layer': 'conv1d',
                'num_filters': 32,
                'filter_size': 6, # 195
                'padding': 'VALID',
                'norm': 'batch',
                'activation': 'relu',
                'max_pool': 3,  # 65
                'dropout': 0.2,
                }
    layer4 = {  'layer': 'conv1d',
                'num_filters': 48,
                'filter_size': 6, # 60
                'padding': 'VALID',
                'norm': 'batch',
                'activation': 'relu',
                'max_pool': 4, # 15
                'dropout': 0.3,
                }
    layer5 = {  'layer': 'conv1d',
                'num_filters': 64,
                'filter_size': 4, # 12
                'padding': 'VALID',
                'norm': 'batch',
                'activation': 'relu',
                'max_pool': 3, # 4
                'dropout': 0.4,
                }
    layer6 = {  'layer': 'conv1d',
                'num_filters': 96,
                'filter_size': 4,
                'padding': 'VALID',
                'norm': 'batch',
                'activation': 'relu',
                'dropout': 0.5,
                }
    layer7 = {  'layer': 'dense',
                'num_units': output_shape[1],
                'activation': 'sigmoid',
                }

    model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.001,
                  "l2": 1e-6,
                  }

    return model_layers, optimization
