import numpy as np

def heUniform(shape):
    fan_in = shape[1]
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)
