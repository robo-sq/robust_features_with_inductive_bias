
def model(input_shape, output_shape, dropout=False, l2=False, batch_norm=False):

    layer1 = {  'layer': 'input',
                'input_shape': input_shape
             }
    layer2 = {  'layer': 'conv1d',
                'num_filters': 24,
                'filter_size': 7,
                'padding': 'SAME',
                'activation': 'relu',
                }
    if dropout:
        layer2['dropout'] = 0.2
    if batch_norm:
        layer2['norm'] = 'batch'
    layer3 = {  'layer': 'conv1d',
                'num_filters': 32,
                'filter_size': 6, # 195
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 3,  # 65
                }
    if dropout:
        layer3['dropout'] = 0.2
    if batch_norm:
        layer3['norm'] = 'batch'

    layer4 = {  'layer': 'conv1d',
                'num_filters': 48,
                'filter_size': 6, # 60
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 4, # 15
                }
    if dropout:
        layer4['dropout'] = 0.3
    if batch_norm:
        layer4['norm'] = 'batch'

    layer5 = {  'layer': 'conv1d',
                'num_filters': 64,
                'filter_size': 4, # 12
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 3, # 3
                }
    if dropout:
        layer5['dropout'] = 0.4
    if batch_norm:
        layer5['norm'] = 'batch'

    layer6 = {  'layer': 'conv1d',
            'num_filters': 96,
            'filter_size': 4,
            'padding': 'VALID',
            'activation': 'relu',
            }
    if dropout:
        layer6['dropout'] = 0.5
    if batch_norm:
        layer6['norm'] = 'batch'

    layer7 = {  'layer': 'dense',
                'num_units': output_shape[1],
                'activation': 'sigmoid',
                }

    model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.0003,
                  }
    if l2:
        optimization['l2'] = 1e-6

    return model_layers, optimization
