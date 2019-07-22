
def model(input_shape, output_shape):

    layer1 = {  'layer': 'input',
                'input_shape': input_shape
             }
    layer2 = {  'layer': 'conv1d',
                'num_filters': 30,
                'filter_size': 7,
                'padding': 'SAME',
                'activation': 'relu',
                }
    layer3 = {  'layer': 'conv1d',
                'num_filters': 48,
                'filter_size': 6, # 195
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 3,  # 65
                }
    layer4 = {  'layer': 'conv1d',
                'num_filters': 64,
                'filter_size': 6, # 60
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 4, # 15
                }
    layer5 = {  'layer': 'conv1d',
                'num_filters': 96,
                'filter_size': 4, # 12
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 3, # 3
                }
    layer6 = {  'layer': 'conv1d',
                'num_filters': 128,
                'filter_size': 4,
                'padding': 'VALID',
                'activation': 'relu',
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
                  }

    return model_layers, optimization
