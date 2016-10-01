import math


def sigmoid(x):
    return float(1) / (1 + math.exp(-x))