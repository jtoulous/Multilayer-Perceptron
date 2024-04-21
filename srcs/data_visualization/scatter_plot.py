import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse as ap

from colorama import Fore, Style

def printLog(message):
    print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")

def printError(message):
    print(f"{Fore.LIGHTRED_EX}{message}{Style.RESET_ALL}")

def printInfo(message):
    print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")


def parseArgs():
    parser = ap.ArgumentParser(
        prog='Scatter Plots',
        description='display scatter plots for csv dataset'
    )
    parser.add_argument('dataFile', help='the csv data file')
    parser.add_argument('-head', '--head', action='store_true', help='if the csv file has a header')
    parser.add_argument('-l', '--label', type=int, help="labels column index")
    return parser.parse_args()


def getTargets(dataset):
    numericalFeatures = list(dataset.select_dtypes(include=['float64']).columns)
    aquiredTarget = 0
    target1 = None
    target2 = None

    while aquiredTarget != 1:
        for i, feature in enumerate(numericalFeatures):
            printInfo(f'{i}: feature {feature}')
        try:
            answer = int(input(f'{Fore.GREEN}Select your first feature: {Style.RESET_ALL}'))
            if answer in range(len(numericalFeatures)):
                target1 = numericalFeatures[answer]
                aquiredTarget += 1
            else:
                printError('Unavailable feature, try again')

        except Exception:
            printError('selected choice must be the number of a feature')

    numericalFeatures.remove(target1)

    while aquiredTarget != 2:
        for i, feature in enumerate(numericalFeatures):
            printInfo(f'{i}: feature {feature}')
        try:
            answer = int(input(f'{Fore.GREEN}Select your second feature: {Style.RESET_ALL}'))
            if answer in range(len(numericalFeatures)):
                target2 = numericalFeatures[answer]
                aquiredTarget += 1
            else:
                printError('Unavailable feature, try again')

        except Exception:
            printError('selected choice must be the number of a feature')
    return target1, target2


def getPossibleLabels(labelsData):
    labels = []
    for data in labelsData:
        if data not in labels:
            labels.append(data)
    return labels


def displayScatter(dataset, target1, target2):
    plt.scatter(dataset[target1], dataset[target2], alpha=0.3, color='darkred')
    plt.xlabel(target1)
    plt.ylabel(target2)
    plt.legend()
    plt.show()


def displayScatterLabels(dataset, feature1, feature2, labelsPossible, labelsIndex):
    colors = ['darkred', 'turquoise', 'black', 'green', 'yellow', 'purple', 'brown', 'orange', 'pink', 'blue', 'gray', 'cyan', 'magenta', 'lime']

    for i, label in enumerate(labelsPossible):
        data_feat1 = dataset.loc[dataset[labelsIndex] == label, feature1]
        data_feat2 = dataset.loc[dataset[labelsIndex] == label, feature2]

        plt.scatter(data_feat1, data_feat2, alpha=0.3, label=label, color=colors[i])
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:
        args = parseArgs()
        dataset = pd.read_csv(args.dataFile, header=None) if args.head == False else pd.read_csv(args.dataFile)        
        target1, target2 = getTargets(dataset)
        labelsPossible = []

        if args.label is not None:
            labelsPossible = getPossibleLabels(dataset[args.label])
            displayScatterLabels(dataset, target1, target2, labelsPossible, args.label)
        else:
            displayScatter(dataset, target1, target2)
        

    except Exception as error:
        printError(error)














