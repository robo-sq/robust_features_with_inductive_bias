
def model(input_shape, output_shape, dropout=False, l2=False, batch_norm=False):

    layer1 = {  'layer': 'input',
                'input_shape': input_shape # 1000
             }
    layer2 = {  'layer': 'conv1d',
                'num_filters': 24,
                'filter_size': 19,
                'padding': 'SAME',
                'activation': 'relu',
                'max_pool': 50, 
                }
    if dropout:
        layer2['dropout'] = 0.2
    if batch_norm:
        layer2['norm'] = 'batch'
    layer3 = {  'layer': 'conv1d',
                'num_filters': 96,
                'filter_size': 4,
                'padding': 'VALID',
                'activation': 'relu',
                }
    if dropout:
        layer3['dropout'] = 0.5
    if batch_norm:
        layer3['norm'] = 'batch'
    layer4 = {  'layer': 'dense',
                'num_units': output_shape[1],
                'activation': 'sigmoid',
                }

    model_layers = [layer1, layer2, layer3, layer4]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.0003,
                  }
    if l2:
        optimization['l2'] = 1e-6

    return model_layers, optimization
