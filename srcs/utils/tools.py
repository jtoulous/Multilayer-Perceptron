import pandas as pd
import copy

from statistics import mean, stdev
from colorama import Fore, Style

class Data:
    def __init__(self, data_train, data_valid, normData, features):
        self.data_train = copy.deepcopy(data_train)
        self.data_valid = copy.deepcopy(data_valid)
        self.normData = copy.deepcopy(normData)
        self.features = copy.deepcopy(features)

def printLog(message):
    print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")

def printError(message):
    print(f"{Fore.LIGHTRED_EX}{message}{Style.RESET_ALL}")

def printInfo(message):
    print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")

def newCmd():
    print ('\n======================================================\n')

def printNetwork(network):
    for i, layer in enumerate(network.layers):
        print(f'layer {i}:\n'
              f'   activation = {layer.activation}\n'
              f'   weight initializer = {layer.weights_initializer}\n'
              f'   type = {layer.type}\n'
              f'   neurons = {len(layer.neurons)}\n'
              f'   prev layer shape = {layer.prevLayerShape}\n'
              f'   shape = {layer.shape}\n')


def normalize(mean, std, value):
    return (value - mean) / std

def denormalize(mean, std, value):
    return (value * std) + mean

def normalizeData(features, *dataframes):
    normData = {'means': {}, 'stds': {}}
    for feature in features:
        fullSet = []
        for dataframe in dataframes:
            fullSet.extend(list(dataframe[feature]))
        normData['means'][feature] = mean(fullSet)
        normData['stds'][feature] = stdev(fullSet)

    for feature in features:
        me = normData['means'][feature]
        std = normData['stds'][feature]
        for dataframe in dataframes:
            for i in range(len(dataframe[feature])):
                dataframe.loc[i, feature] = normalize(me, std, dataframe[feature][i])

    return normData


def cleanData(featuresToUse, *dataframes):
    for dataframe in dataframes:    
        for feature in featuresToUse:
            median = dataframe[feature].median()
            dataframe.loc[dataframe[feature].isna(), feature] = median  

def getFeaturesToDrop(features):
    featuresToUse = []
    featuresToDrop = []
    featuresCopy = features.copy()

    while (1):
        newCmd()
        printInfo('Available features:\n')
        for i, feature in enumerate(features):
            printInfo(f'{i}: feature {feature}')
        printInfo('\nall: all features')
        printInfo('done: finished\n')
        
        answer = input(f'{Fore.GREEN}Select a feature: {Style.RESET_ALL}')
        if answer == 'all':
            featuresToUse = featuresCopy.copy()
        if answer == 'done' or answer == 'all':
            break
        else:
            try:
                answer = int(answer)
                if answer in range(len(features)):
                    featuresToUse.append(features[answer])
                    features.pop(answer)
            except Exception:
                printError('Selected feature as to be a valid number')

    for feature in features:
        if feature not in featuresToUse:
            featuresToDrop.append(feature)

    return featuresToUse, featuresToDrop


def getData():
    data_train = []
    data_valid = []
    try:
        df_train = pd.read_csv('datasets/training_data.csv', header=None)
        df_valid = pd.read_csv('datasets/validation_data.csv', header=None)
    except Exception:
        raise Exception('Error: you need to separate the data before training')
    
    featuresToUse, featuresToDrop = getFeaturesToDrop(list(df_train.columns[2:]))
    df_train = df_train.drop(columns=featuresToDrop)
    df_valid = df_valid.drop(columns=featuresToDrop)
    
    cleanData(featuresToUse, df_train, df_valid)
    normData = normalizeData(featuresToUse, df_train, df_valid)

    for i in range(len(df_train)):
        newData = {'id': df_train[0][i], 'label': df_train[1][i], 'features': {}}
        for feature in featuresToUse:
            newData['features'][feature] = df_train[feature][i]
        data_train.append(newData)

    for i in range(len(df_valid)):
        newData = {'id': df_valid[0][i], 'label': df_valid[1][i], 'features': {}}
        for feature in featuresToUse:
            newData['features'][feature] = df_valid[feature][i]
        data_valid.append(newData)

    return Data(data_train, data_valid, normData, featuresToUse)

def getLabels(*dataframes):
    fullSet = []
    labelsEncountered = []

    for dataframe in dataframes:
        fullSet.extend(dataframe)

    for data in fullSet:
        if data['label'] not in labelsEncountered:
            labelsEncountered.append(data['label'])
    return labelsEncountered