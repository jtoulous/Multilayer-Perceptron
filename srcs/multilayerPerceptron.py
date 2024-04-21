import sys
import pandas as pd
import argparse as ap

from colorama import Fore, Style
from utils.logs import printError, printLog, printInfo

def getArgs():
    parser = ap.ArgumentParser(
        prog='Multilayer Perceptron',
        description='training model to detect malignant or benin tumors',
        )
    parser.add_argument('dataFile', help='the csv data file')
    return parser.parse_args()


def splitData(dataFile):
    printInfo('Running splitter...')
    dataset = pd.read_csv(dataFile, header=None)
    dataset = dataset.sample(frac=1, random_state=42)

    nb_trainingData = int((len(dataset) * 80) / 100)
    
    training_data = dataset.iloc[:nb_trainingData]
    test_data = dataset.iloc[nb_trainingData:]

    training_data.to_csv('utils/datasets/training_data.csv', index=False, header=False)
    test_data.to_csv('utils/datasets/validation_data.csv', index=False, header=False)
    printInfo('Done')


def training():


if __name__ == '__main__':
    try:
        args = getArgs()
        validChoice = 0

        while (validChoice == 0):
            validChoice = 1    
            printInfo('Program choice:\n  1- Dataset splitter\n  2- Training program\n  3- Prediction program')
            progToUse = input(f'{Fore.GREEN}\nSelect a program: {Style.RESET_ALL}')
            
            if progToUse == "1":
                splitData(args.dataFile)
            elif progToUse == "2":
                training()
            #elif progToUse == "3":
            #    prediction()
            else:
                validChoice = 0

    except Exception as error:
        printError(f'{error}')
