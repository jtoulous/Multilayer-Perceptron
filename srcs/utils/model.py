from .weight_initializer import heUniform

class Model:
    @staticmethod
    def createNetwork(layers):
        return Network(layers)

    @staticmethod
    def fit(network, data, loss, learning_rate, batch_size, epochs):
        cost_history = []
        
        for epoch in range(epochs):
            batches = getBatches(data.data_train)
            
            for batch in batches:
                for layer in network.layers:
                    batch = activateNeurons(layer, batch)

                cost_history.append(getCost(loss))
                retropropagation(network, batch)




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
            weightsList = list(heUniform([self.shape, self.prevLayerShape + 1]))     
        for i, neuron in enumerate(self.neurons):
            weights = list(weightsList[i])
            neuron.bias = weights.pop()
            neuron.weights = weights.copy()

    @staticmethod
    def DenseLayer(shape, activation='sigmoid', weights_initializer='heUniform'):
        return Layers(shape, activation, weights_initializer)


class Neuron:
    def __init__(self):
        self.weights = []
        self.bias = None
        self.activationResults = None
        self.scores