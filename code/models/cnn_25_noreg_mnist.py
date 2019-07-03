
def model(input_shape, output_shape):

    layer1 = {  'layer': 'input',
                'input_shape': input_shape # 1000
             }
    layer2 = {  'layer': 'conv2d',
                'num_filters': 30,
                'filter_size': 19,
                'padding': 'SAME',
                'activation': 'relu',
                'max_pool': 25, # 4
                }
    layer3 = {  'layer': 'conv2d',
               'num_filters': 64,
               'filter_size': 5,
               'padding': 'SAME',
               'activation': 'relu',
               'max_pool': 4, # 4
               }
    layer4 = {  'layer': 'dense',
               'num_units': 128,
               'activation': 'relu',
               }
    layer5 = {  'layer': 'dense',
                'num_units': output_shape[1],
                'activation': 'sigmoid',
                }

    model_layers = [layer1, layer2, layer3, layer4, layer5]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.001,
                  }

    return model_layers, optimization
