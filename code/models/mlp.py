
def model(input_shape, output_shape):

    layer1 = {  'layer': 'input',
                'input_shape': input_shape # 1000
             }
    layer2 = {  'layer': 'dense',
                'num_units': 64,
                'norm': 'batch',
                'dropout': 0.1,
                'activation': 'relu',
                }
    layer3 = {  'layer': 'dense',
                'num_units': 128,
                'norm': 'batch',
                'dropout': 0.1,
                'activation': 'relu',
                }
    layer4 = {  'layer': 'dense',
                'num_units': output_shape[1],
                'activation': 'sigmoid',
                }

    model_layers = [layer1, layer2, layer3, layer4]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.001,
                  "l2": 1e-6,
                  }

    return model_layers, optimization
