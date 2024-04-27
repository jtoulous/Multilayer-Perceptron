import math

def calcScore(data, weights, bias):
    score = 0
    for i, feature in enumerate(data):
        score += (weights[feature] * data[feature])
    return score + bias


def sigmoid(score):
    return 1 / (1 + math.exp(-score))


def softmax(value, *scores):
    totalScores = sum(math.exp(score) for score in scores)
    return math.exp(value) / totalScores
