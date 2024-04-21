from colorama import Fore, Style


def printLog(message):
    print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")

def printError(message):
    print(f"{Fore.LIGHTRED_EX}{message}{Style.RESET_ALL}")

def printInfo(message):
    print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")
