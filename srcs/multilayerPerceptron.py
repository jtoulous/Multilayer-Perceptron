import sys
import os
import pandas as pd
import argparse as ap
import numpy as np
import random

from colorama import Fore, Style
from utils.model import Model, Network, Layers
from utils.tools import printError, printLog, printInfo, getData, printNetwork, getLabels, printNeuron#, getConfig

def parsing():
    parser = ap.ArgumentParser(
        prog='Multilayer Perceptron',
        description='training model to detect malignant or benin tumors',
        )
    parser.add_argument('dataFile', help='the csv data file')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='the number of epochs')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.1, help='the learning rate')
    parser.add_argument('-l', '--loss', default='binaryCrossentropy', help='the loss fonction')
    parser.add_argument('-b', '--batchs', type=int, default=10, help='the batchs size')
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24], help='the number of neurons in as many hidden layers desired')
    return parser.parse_args()


def splitData(dataFile):
    printInfo('Running splitter...')
    dataset = pd.read_csv(dataFile, header=None)
    dataset = dataset.sample(frac=1, random_state=42)

    nb_trainingData = int((len(dataset) * 80) / 100)
    training_data = dataset.iloc[:nb_trainingData]
    validation_data = dataset.iloc[nb_trainingData:]

    training_data.to_csv('datasets/training_data.csv', index=False, header=False)
    validation_data.to_csv('datasets/validation_data.csv', index=False, header=False)
    printInfo('Done')


def training(args):
    data = getData()
    #input_shape = [len(data.features), 30]
    input_shape = len(data.features)
    output_shape = len(getLabels(data.data_train, data.data_valid))

    network = Model.createNetwork([
        Layers.DenseLayer(input_shape, activation='sigmoid'),
        *Layers.HiddenLayers(args.layer),
        Layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
    ], data)
    Model.fit(network, data, loss=args.loss, learning_rate=args.learning_rate, batch_size=args.batchs, epochs=args.epochs)


def prediction(datafile):
#    network, normData, dataset = getConfig(datafile)
    dataframe = pd.read_csv(datafile, header=None)
    dataset = []
    features = []
    features_to_drop = []
    layers = []
    architecture = []
    network = None
    normData = {'means': {}, 'stds': {}}

    if not os.path.exists('utils/network.txt'):
        raise Exception('Error: run training before running predictions')
    
    with open('utils/network.txt', 'r') as network_file:
        features_line = network_file.readline().split(':')[1]
        features = [feat.strip() for feat in features_line.split(',')]
        
        architecture_line = network_file.readline().split(':')[1]
        architecture = [layer.strip() for layer in architecture_line.split(',')]
        for layer_info in architecture:
            shape, activation, initializer = layer_info.split('|')
            layers.append(Layers.DenseLayer(int(shape), activation, initializer))
        
        network = Model.createNetwork(layers)        
        for layer in network.layers:
            network_file.readline()
            for neuron in layer.neurons:
                neuron_line = network_file.readline()
                if layer.type == 'output':
                    neuron.label = neuron_line.split(':')[0]
                neuron_line = neuron_line.split(':')[1]
                neuron.bias = float(neuron_line.split('|')[1])
                neuron_line = neuron_line.split('|')[0]
                for weight_info in neuron_line.split(','):
                    neuron.weights[weight_info.split('=')[0]] = float(weight_info.split('=')[1])

        means_line = network_file.readline().split(':')[1]
        for mean in means_line.split(','):
            mean.strip()
            normData['means'][mean.split('=')[0]] = float(mean.split('=')[1])
        
        stds_line = network_file.readline().split(':')[1]
        for std in stds_line.split(','):
            std.strip()
            normData['stds'][std.split('=')[0]] = float(std.split('=')[1])
    
    ########  A DEBUGGER  #######
    for column in dataframe.columns:
        if column not in features:
            features_to_drop.append(column)
    dataframe.drop(features_to_drop)
    for i in range(len(dataframe)):
        new_data = {'id': dataframe[0][i], 'label': dataframe[1][i], 'features': {}}
        for feature in features:
            new_data['features'][feature] = dataframe[feature][i]
        dataset.append(new_data)
    breakpoint()


if __name__ == '__main__':
    try:
        args = parsing()
        validChoice = 0

        while (validChoice == 0):
            validChoice = 1    
            printInfo('Program choice:\n  1- Dataset splitter\n  2- Training program\n  3- Prediction program')
            progToUse = input(f'{Fore.GREEN}\nSelect a program: {Style.RESET_ALL}')
            
            if progToUse == "1":
                splitData(args.dataFile)
            elif progToUse == "2":
                training(args)
            elif progToUse == "3":
                prediction(args.dataFile)
            else:
                validChoice = 0

    except Exception as error:
        printError(f'{error}')
