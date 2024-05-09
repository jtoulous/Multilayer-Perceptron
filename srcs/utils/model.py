import copy
import random
import sys

from colorama import Style, Fore
from statistics import mean
from .weight_initializer import heUniform
from .activation import calcScore, sigmoid, softmax
from .cost import getMeanCost
from .tools import getLabels, printNetwork, printNeuron, printError, printInfo, printLog

class Model:
    @staticmethod
    def createNetwork(layers, features):
        return Network(layers, features)

    @staticmethod
    def fit(network, data, loss, learning_rate, batch_size, epochs):
        bestNetworkConfig = copy.deepcopy(network)
        meanCostHistory = []
        for epoch in range(epochs):
            totalCosts = []
            batches = getBatches(data.data_train, batch_size)

            for batch in batches:
                tmp_batch = copy.deepcopy(batch)
                for layer in network.layers:
                    tmp_batch = activateNeurons(layer, tmp_batch)
                totalCosts.append(getMeanCost(loss, tmp_batch))
                retropropagation(network, batch, tmp_batch, loss, learning_rate)
#                validation()
                network.resetNetwork()
            
            meanCost = mean(totalCosts)
            meanCostHistory.append(mean(totalCosts))
            bestNetworkConfig = copy.deepcopy(network) if meanCost == min(meanCostHistory) else bestNetworkConfig
            print(f'Epoch {epoch}: {meanCost}')
#        saveConfig(bestNetworkConfig)
        printPredictions(bestNetworkConfig, data.data_train, data.data_valid)
#        printGraphs(meanCostHistory, )


class Network:
    def __init__(self, layers, data):
        self.layers = layers.copy()
        for i, layer in enumerate(self.layers):
            if layer.prevLayerShape is None:
                layer.prevLayerShape = layers[i - 1].shape
            layer.type = 'input' if i == 0 else 'output' if i == len(self.layers) - 1 else 'hidden'
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
            neuron.errors = None

    @staticmethod
    def DenseLayer(shape, activation='sigmoid', weights_initializer='heUniform'):
        return Layers(shape, activation, weights_initializer)



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
    value_list = []
    keys_list = list(batch[0]['features'].keys())
    
    for data in batch:
        features_values = data['features']
        value_list.append(features_values[keys_list[index]] * error)
    return mean(value_list)

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