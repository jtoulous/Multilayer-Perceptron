import copy
import random
import sys

from colorama import Style, Fore
from statistics import mean
from .weight_initializer import heUniform
from .activation import calcScore, sigmoid, softmax
from .cost import getMeanCost
from .tools import getLabels, printError, printInfo
from .tools import printLog, printGraphs, printEpochResult, printDataShapes, saveConfig

class Model:
    @staticmethod
    def createNetwork(layers, data=None):
        return Network(layers, data)

    @staticmethod
    def fit(network, data, loss, learning_rate, batch_size, epochs):
        bestNetworkConfig = copy.deepcopy(network)
        meanCostHistory = {'train data': [], 'valid data': []}
        precisionHistory = {'train data': [], 'valid data': []}
        printDataShapes(data)
        for epoch in range(epochs):
            totalCosts = []
            batches = getBatches(data.data_train, batch_size)

            for batch in batches:
                tmp_batch = copy.deepcopy(batch)
                for layer in network.layers:
                    tmp_batch = activateNeurons(layer, tmp_batch)
                totalCosts.append(getMeanCost(loss, tmp_batch))
                retropropagation(network, batch, tmp_batch, loss, learning_rate)
                network.resetNetwork()

            meanCostValid = validation(network, data.data_valid, loss)
            meanCostTrain = mean(totalCosts)
            meanCostHistory['train data'].append(meanCostTrain)
            meanCostHistory['valid data'].append(meanCostValid)
            addPrecision(precisionHistory, network, data)

            bestNetworkConfig = copy.deepcopy(network) if meanCostTrain == min(meanCostHistory['train data']) else bestNetworkConfig
            printEpochResult(epoch, epochs, meanCostTrain, meanCostValid)

        saveConfig(bestNetworkConfig, data)
        printPredictions(bestNetworkConfig, data.data_train, data.data_valid)
        printGraphs(meanCostHistory, precisionHistory)


class Network:
    def __init__(self, layers, data):
        self.layers = layers.copy()
        for i, layer in enumerate(self.layers):
            layer.prevLayerShape = layers[i - 1].shape if i != 0 else layer.shape
            layer.type = 'input' if i == 0 else 'output' if i == len(self.layers) - 1 else 'hidden'
            if data != None:    
                if layer.type == 'output':
                    self.defineOutputNeurons(layer, data)
                layer.initWeights(data.features)
            
    def defineOutputNeurons(self, layer, data):
        labels = getLabels(data.data_train, data.data_valid)
        for i, label in enumerate(labels):
            layer.neurons[i].label = label

    def resetNetwork(self):
        for layer in self.layers:
            layer.cleanNeurons()



class Layers:
    def __init__(self, shape, activation, weights_initializer):
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.type = None
        self.neurons = [Neuron() for i in range(shape)] 
        self.shape = shape

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
            neuron.errors = None

    @staticmethod
    def DenseLayer(shape, activation='sigmoid', weights_initializer='heUniform'):
        return Layers(shape, activation, weights_initializer)

    @staticmethod
    def HiddenLayers(layers):
        layers_list = []
        for layer_shape in layers:
            layers_list.append(Layers.DenseLayer(layer_shape, activation='sigmoid', weights_initializer='heUniform'))
        return layers_list


class Neuron:
    def __init__(self):
        self.weights = {}
        self.bias = None
        self.scores = []
        self.activationResults = []
        self.label = None
        self.errors = None


###############################################################################


def getBatches(dataset, batch_size):
    dataset_tmp = copy.deepcopy(dataset)
    random.shuffle(dataset_tmp)
    batches = [dataset_tmp[i:i + batch_size] for i in range(0, len(dataset_tmp), batch_size)]
    return batches


def activateNeurons(layer, dataset):
    new_dataset = []

    for data in dataset:
        for neuron in layer.neurons:
            neuron.scores.append(calcScore(data['features'], neuron.weights, neuron.bias))
    
    for i, data in enumerate(dataset):
        new_data = {'id': data['id'], 'label': data['label'], 'features': {}}
        for j, neuron in enumerate(layer.neurons):
            if layer.activation == "sigmoid":
                activation_res = sigmoid(neuron.scores[i])
            elif layer.activation == "softmax":
                activation_res = softmax(neuron.scores[i], *[nrn.scores[i] for nrn in layer.neurons])
            else:
                raise Exception(f'Error: {layer.activation} not available')  
            
            neuron.activationResults.append(activation_res)
            if neuron.label is None:
                new_data['features'][str(j)] = activation_res
            else:
                new_data['features'][neuron.label] = activation_res
        new_dataset.append(new_data)

    return new_dataset


def retropropagation(network, batch, tmp_batch, loss, learning_rate):
    for l in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[l]
        prev_layer = network.layers[l - 1] if l != 0 else None
        next_layer = network.layers[l + 1] if l != len(network.layers) - 1 else None
      
        if l == len(network.layers) - 1:
            for neuron in layer.neurons:
                neuron.errors = getMeanCost(loss, tmp_batch, neuron.label, retropropagation=True)
                for w in range(len(neuron.weights)):
                    neuron.weights[str(w)] = neuron.weights[str(w)] - learning_rate * getMeanGradient(neuron.errors, prev_layer.neurons[w].activationResults)
                neuron.bias = neuron.bias - learning_rate * neuron.errors

        elif l != 0:
            for i, neuron in enumerate(layer.neurons):
                totalError = []
                weighted_sum = 0
                for n in next_layer.neurons:
                    weighted_sum += n.weights[str(i)] * n.errors
                for activation in neuron.activationResults:
                    derived_activation = activation * (1 - activation)
                    totalError.append(weighted_sum * derived_activation)
                neuron.errors = mean(totalError)
                for w, weight in enumerate(neuron.weights.keys()):
                    neuron.weights[weight] = neuron.weights[weight] - learning_rate * getMeanGradient(neuron.errors, prev_layer.neurons[w].activationResults)
                neuron.bias = neuron.bias - learning_rate * neuron.errors

        else:
            for i, neuron in enumerate(layer.neurons):
                totalError = []
                weighted_sum = 0
                for n in next_layer.neurons:
                    weighted_sum += n.weights[str(i)] * n.errors
                for activation in neuron.activationResults:
                    derived_activation = activation * (1 - activation)
                    totalError.append(weighted_sum * derived_activation)
                neuron.errors = mean(totalError)
                for w, weight in enumerate(neuron.weights.keys()):
                    neuron.weights[weight] = neuron.weights[weight] - learning_rate * getMeanGradientInput(w, batch, neuron.errors)
                neuron.bias = neuron.bias - learning_rate * neuron.errors


def getMeanGradient(neuron_error, activation_results):
    totalGradients = []
    for activation in activation_results:
        totalGradients.append(neuron_error * activation)
    return mean(totalGradients)   


def getMeanGradientInput(index, batch, error):
    totalGradients = []
    keys_list = list(batch[0]['features'].keys())
    
    for data in batch:
        features_values = data['features']
        totalGradients.append(features_values[keys_list[index]] * error)
    return mean(totalGradients)


def printPredictions(bestNetworkConfig, *datasets):
    full_dataset = []
    correct_count = 0

    for dataset in datasets:
        full_dataset.extend(dataset)
    
    for layer in bestNetworkConfig.layers:
        full_dataset = activateNeurons(layer, full_dataset)
    
    printInfo('Predictions:\n')
    for data in full_dataset:
        prediction = 'Benin' if data['features']['B'] > data['features']['M'] else 'Malignant'
        tumor_type = 'Malignant' if data['label'] == 'M' else 'Benin'
        if prediction == tumor_type:
            printLog(f'ID {data["id"]}: {tumor_type} ====> {prediction}')
            correct_count += 1
        else:
            printError(f'ID {data["id"]}: {tumor_type} ====> {prediction}')
    printLog(f'\n{int((correct_count / len(full_dataset)) * 100)}% successfull predictions\n')


def validation(network, dataset, loss):
    tmp_data = copy.deepcopy(dataset)
    tmp_network = copy.deepcopy(network)

    for layer in tmp_network.layers:
        tmp_data = activateNeurons(layer, tmp_data)
    return getMeanCost(loss, tmp_data)


def addPrecision(precisionHistory, network, data):
    data_training = copy.deepcopy(data.data_train)
    data_validation = copy.deepcopy(data.data_valid)

    tmp_network = copy.deepcopy(network)
    correct_count = 0
    for layer in tmp_network.layers:
        data_training = activateNeurons(layer, data_training)
    for elem in data_training:
        prediction = 'M' if elem['features']['M'] > elem['features']['B'] else 'B'
        if elem['label'] == prediction:
            correct_count += 1
    precisionHistory['train data'].append(correct_count / len(data.data_train))

    tmp_network = copy.deepcopy(network)
    correct_count = 0
    for layer in tmp_network.layers:
        data_validation = activateNeurons(layer, data_validation)
    for elem in data_validation:
        prediction = 'M' if elem['features']['M'] > elem['features']['B'] else 'B'
        if elem['label'] == prediction:
            correct_count += 1
    precisionHistory['valid data'].append(correct_count / len(data.data_valid))