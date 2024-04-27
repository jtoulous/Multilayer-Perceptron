import math

def calcScore(data, weights, bias):
    score = 0
    for i, feature in enumerate(data):
        score += (weights[feature] * data[feature])
    return score + bias


def calcActivation(score, activation): ################  REFAIRE
    if activation == "sigmoid":
        return sigmoid(score)
    elif activation == "softmax":
        return softmax()

    else:
        raise Exception(f'Error: {activation} not available')


def sigmoid(score):
    return 1 / (1 + math.exp(-score))