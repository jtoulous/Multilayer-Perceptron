import copy
import random

from .weight_initializer import heUniform
from .activation import calcScore, calcActivation

class Model:
    @staticmethod
    def createNetwork(layers, features):
        return Network(layers, features)

    @staticmethod
    def fit(network, data, loss, learning_rate, batch_size, epochs):
        meanCostHistory = []
        for epoch in range(epochs):
            totalCosts = []
            batches = getBatches(data.data_train, batch_size)

            for batch in batches:
                for layer in network.layers:
                    batch = activateNeurons(layer, batch)
                    breakpoint()

                totalCosts.append(getCost(loss))
                retropropagation(network, batch)
                network.resetNetwork()




class Network:
    def __init__(self, layers, features):
        self.layers = layers.copy()
        for i, layer in enumerate(self.layers):
            if layer.prevLayerShape is None:
                layer.prevLayerShape = layers[i - 1].shape
            layer.type = 'input' if i == 0 else 'output' if i == len(self.layers) - 1 else 'hidden'
            layer.initWeights(features)

    def resetNetwork(self):
        for layer in self.layers:
            layer.cleanNeurons()


class Layers:
    def __init__(self, shape, activation, weights_initializer):
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.type = None
        self.neurons = [Neuron() for i in range(shape[1])] if isinstance(shape, list) else [Neuron() for i in range(shape)] 
        self.shape = shape[1] if isinstance(shape, list) else shape
        self.prevLayerShape = shape[0] if isinstance(shape, list) else None

    def initWeights(self, features):
        if self.weights_initializer == 'heUniform':
            weightsList = list(heUniform([self.shape, self.prevLayerShape + 1]))    
        for i, weights in enumerate(weightsList):
            weights = list(weights)
            self.neurons[i].bias = weights.pop()
            if self.type == 'input':
                self.neurons[i].weights = {feature: weights[j] for j, feature in enumerate(features)}
            else:
                self.neurons[i].weights = {str(j): weights[j] for j in range(len(weights))}

    def cleanNeurons(self):
        for neuron in self.neurons:
            neuron.scores = []
            neuron.activationResults = []

    @staticmethod
    def DenseLayer(shape, activation='sigmoid', weights_initializer='heUniform'):
        return Layers(shape, activation, weights_initializer)


class Neuron:
    def __init__(self):
        self.weights = {}
        self.bias = None
        self.scores = []
        self.activationResults = []


###############################################################################


def getBatches(dataset, batch_size):
    dataset_tmp = copy.deepcopy(dataset)
    random.shuffle(dataset_tmp)
    batches = [dataset_tmp[i:i + batch_size] for i in range(0, len(dataset_tmp), batch_size)]
    return batches


def activateNeurons(layer, dataset):#####################################    REFAIRE
    new_dataset = []

    for data in dataset:
        new_data = {'id': data['id'], 'label': data['label'], 'features': {}}
        
        for i, neuron in enumerate(layer.neurons):
            score = calcScore(data['features'], neuron.weights, neuron.bias)
            activation_res = calcActivation(score, layer.activation)

            neuron.scores.append(score)
            neuron.activationResults.append(activation_res)
            new_data['features'][str(i)] = activation_res
        new_dataset.append(new_data)
    
    return new_dataset












