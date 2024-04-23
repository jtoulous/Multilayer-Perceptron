import sys
import pandas as pd
import argparse as ap

from colorama import Fore, Style
from utils.model import Model, Network, Layers
from utils.tools import printError, printLog, printInfo, getData

def printNetwork(network):
    for i, layer in enumerate(network.layers):
        print(f'layer {i}:\n'
              f'   activation = {layer.activation}\n'
              f'   weight initializer = {layer.weights_initializer}\n'
              f'   type = {layer.type}\n'
              f'   neurons = {len(layer.neurons)}\n'
              f'   prev layer shape = {layer.prevLayerShape}\n'
              f'   shape = {layer.shape}\n')

def parsing():
    parser = ap.ArgumentParser(
        prog='Multilayer Perceptron',
        description='training model to detect malignant or benin tumors',
        )
    parser.add_argument('dataFile', help='the csv data file')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.1, help='the learning rate')
    parser.add_argument('-l', '--loss', default='binaryCrossentropy', help='the loss fonction')
    parser.add_argument('-b', '--batch', type=int, default=10, help='the batchs size')
    return parser.parse_args()


def splitData(dataFile):
    printInfo('Running splitter...')
    dataset = pd.read_csv(dataFile, header=None)
    dataset = dataset.sample(frac=1, random_state=42)

    nb_trainingData = int((len(dataset) * 80) / 100)
    
    training_data = dataset.iloc[:nb_trainingData]
    test_data = dataset.iloc[nb_trainingData:]

    training_data.to_csv('datasets/training_data.csv', index=False, header=False)
    test_data.to_csv('datasets/validation_data.csv', index=False, header=False)
    printInfo('Done')


def training(args):
    data_train, data_valid, normData = getData()
#    input_shape = getInputShape()  ########################     ICI
#    output_shape = getOutputShape()
#
#    network = Model.createNetwork([
#        Layers.DenseLayer(input_shape, activation='sigmoid'),
#        Layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
#        Layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
#        Layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
#        Layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
#    ])
#
#    Model.fit(network, data_train, data_valid, loss='binaryCrossentropy', learning_rate=learningRate, batch_size=batchSize, epochs=epoch)


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
            #elif progToUse == "3":
            #    prediction()
            else:
                validChoice = 0

    except Exception as error:
        printError(f'{error}')
