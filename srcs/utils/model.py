from .weight_initializer import heUniform

class Model:
    @staticmethod
    def createNetwork(layers):
        return Network(layers)

    @staticmethod
    def fit(self):
        return 0


class Network:
    def __init__(self, layers):
        self.layers = layers.copy()
        for i, layer in enumerate(self.layers):
            if layer.prevLayerShape is None:
                layer.prevLayerShape = layers[i - 1].shape
            layer.type = 'input' if i == 0 else 'output' if i == len(self.layers) - 1 else 'hidden'
            layer.initWeights()


class Layers:
    def __init__(self, shape, activation, weights_initializer):
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.type = None
        self.neurons = [Neuron() for i in range(shape[1])] if isinstance(shape, list) else [Neuron() for i in range(shape)] 
        self.shape = shape[1] if isinstance(shape, list) else shape
        self.prevLayerShape = shape[0] if isinstance(shape, list) else None

    def initWeights(self):
        if self.weights_initializer == 'heUniform':
            weightsList = heUniform([self.shape, self.prevLayerShape])     
        for i, neuron in enumerate(self.neurons):
            neuron.weights = weightsList[i].copy()

    @staticmethod
    def DenseLayer(shape, activation='sigmoid', weights_initializer='heUniform'):
        return Layers(shape, activation, weights_initializer)


class Neuron:
    def __init__(self):
        self.weights = []
        self.bias = None
        self.activationResult = None