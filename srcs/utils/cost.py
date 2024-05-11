import numpy as np

from statistics import mean

def getMeanCost(loss_function, dataset, neuron_label=None, retropropagation=False):
    if loss_function == "binaryCrossentropy":
        return binaryCrossEntropy(dataset, neuron_label, retropropagation)
    else:
        raise Exception(f'Error: loss function {loss_function} is not implemented in this program')

def binaryCrossEntropy(dataset, neuron_label, retropropagation):
    errors = []
    epsilon = 1e-9

    if retropropagation is True:
        for data in dataset:
            y = 1 if data['label'] == neuron_label else 0
            prob = data['features'][neuron_label]
            errors.append(prob - y)
        return np.mean(errors)

    else:
        for data in dataset:
            y = 1 if data['label'] == 'M' else 0
            prob = data['features']['M']
            #prob = np.clip(data['features']['M'], epsilon, 1 - epsilon)
            errors.append(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        return -np.mean(errors)