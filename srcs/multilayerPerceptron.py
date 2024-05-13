import sys
import os
import pandas as pd
import argparse as ap
import numpy as np
import random

from colorama import Fore, Style
from utils.model import Model, Network, Layers, activateNeurons
from utils.tools import printError, printLog, printInfo, getData, getLabels, getConfig
from utils.cost import getMeanCost

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

    training_data.to_csv('data/training_data.csv', index=False, header=False)
    validation_data.to_csv('data/validation_data.csv', index=False, header=False)
    printInfo('Done')


def training(args):
    data = getData()
    input_shape = len(data.features)
    output_shape = len(getLabels(data.data_train, data.data_valid))

    network = Model.createNetwork([
        Layers.DenseLayer(input_shape, activation='sigmoid'),
        *Layers.HiddenLayers(args.layer),
        Layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
    ], data)
    Model.fit(network, data, loss=args.loss, learning_rate=args.learning_rate, batch_size=args.batchs, epochs=args.epochs)


def prediction(datafile):
    network, normData, dataset = getConfig(datafile)
    correct_count = 0

    for layer in network.layers:
        dataset = activateNeurons(layer, dataset)
    cost = getMeanCost('binaryCrossentropy', dataset)

    for data in dataset:
        prediction = 'Benin' if data['features']['B'] > data['features']['M'] else 'Malignant'
        tumor_type = 'Benin' if data['label'] == 'B' else 'Malignant'
        if prediction == tumor_type:
            printLog(f'ID {data["id"]}: {tumor_type} ====> {prediction}')
            correct_count += 1
        else:
            printError(f'ID {data["id"]}: {tumor_type} ====> {prediction}')
    printLog(f'\n{int((correct_count / len(dataset)) * 100)}% successfull predictions')
    printLog(f'Mean cost ===> {cost}')


def reset():
    printInfo('Resetting...')
    try:
        os.remove('data/training_data.csv')
    except FileNotFoundError:
        pass
    
    try:    
        os.remove('data/validation_data.csv')
    except FileNotFoundError:
        pass

    try:
        os.remove('data/network.txt')
    except FileNotFoundError:
        pass
    printInfo('Done')


if __name__ == '__main__':
    try:
        args = parsing()
        validChoice = 0

        while (validChoice == 0):
            validChoice = 1    
            printInfo('Program choice:\n  1- Dataset splitter\n  2- Training program\n  3- Prediction program\n  4- Reset')
            progToUse = input(f'{Fore.GREEN}\nSelect a program: {Style.RESET_ALL}')
            
            if progToUse == "1":
                splitData(args.dataFile)
            elif progToUse == "2":
                training(args)
            elif progToUse == "3":
                prediction(args.dataFile)
            elif progToUse == "4":
                reset()
            else:
                validChoice = 0

    except Exception as error:
        printError(f'{error}')
