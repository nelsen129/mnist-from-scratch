import activations
import numpy as np


class NeuralNetworkModel:
    def __init__(self, in_features=784, out_features=10, layers=3, channels=16, activation='relu'):
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = layers
        self.channels = channels
        self.activation = activation

        self.layers = []
        for layer in range(layers):
            in_f = in_features if layer == 0 else channels
            out_f = out_features if layer == layers - 1 else channels
            self.layers.append(self.create_layer(in_f, out_f))

    def create_layer(self, in_features, out_features):
        return np.random.random([in_features + 1, out_features])

    def forward_pass(self, inputs):
        x = inputs
        for layer in self.layers:
            x = np.matmul(x, layer[:-1]) + layer[-1]
            x = activations.activation_dict[self.activation][0](x)
        return x
