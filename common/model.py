import numpy as np
from common import activations
from common import metrics
from math import ceil
from tqdm import tqdm

rng = np.random.default_rng(12345)


class NeuralNetworkModel:
    def __init__(self, in_features=784, out_features=10, layers=3, channels=16, activation='relu', learning_rate=1e-4):
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = layers
        self.channels = channels
        self.activation = activation
        self.learning_rate = learning_rate

        self.layers = []
        for layer in range(layers):
            in_f = in_features if layer == 0 else channels
            out_f = out_features if layer == layers - 1 else channels
            self.layers.append(rng.uniform(-0.2, 0.2, size=[in_f + 1, out_f]))

    def forward_pass(self, inputs):
        x = inputs
        logits = [x]
        for i in range(len(self.layers)):
            x = np.matmul(x, self.layers[i][:-1]) + self.layers[i][-1]
            x = activations.activation_dict[self.activation][0](x)
            logits.append(x)
        return x, logits

    def get_gradient(self, logits, labels):
        pred = logits[-1]
        true = labels
        d_a = metrics.d_mse(true, pred)
        gradient_vector = []
        for i in range(-1, -len(self.layers) - 1, -1):
            layer = self.layers[i]
            pred_i = logits[i-1]
            z = np.matmul(pred_i, layer[:-1]) + layer[-1]
            d_z = activations.activation_dict[self.activation][1](z)
            d_w = (d_a * d_z)[:, np.newaxis, :] * pred_i[:, :, np.newaxis]
            d_b = d_a * d_z
            gradient_vector.insert(0, np.concatenate([d_w, d_b[:, np.newaxis, :]], axis=-2))
            d_a = np.matmul(layer[:-1], (d_a * d_z)[:, :, np.newaxis])[:, :, 0]
        return gradient_vector

    def optimizer_step(self, logits, labels):
        gv = self.get_gradient(logits, labels)
        for layer, gradient in zip(self.layers, gv):
            layer -= gradient.mean(axis=0) * self.learning_rate

    def train_step(self, inputs, labels):
        y, logits = self.forward_pass(inputs)
        self.optimizer_step(logits, labels)
        return y

    def fit(self, inputs, labels, epochs=1, batch_size=64, shuffle=True):
        train_x = inputs
        train_y = labels

        history = {'losses': [], 'metrics': []}
        for epoch in range(epochs):
            print(f'Epoch {epoch}:')

            if shuffle:
                permutation = rng.permutation(len(labels))
                train_x = train_x[permutation]
                train_y = train_y[permutation]

            loss = []
            metric = []
            for batch in tqdm(range(ceil(len(inputs) / batch_size))):
                batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
                batch_y = train_y[batch*batch_size:(batch+1)*batch_size]
                batch_y_pred = self.train_step(batch_x, batch_y)
                loss.append(metrics.mse(batch_y, batch_y_pred))
                metric.append(metrics.categorical_acc(batch_y, batch_y_pred))

            loss = np.concatenate(loss).mean()
            metric = np.array(metric).sum() / len(inputs)

            print(f'Loss: {loss}, Metric: {metric}')

            history['losses'].append(loss)
            history['metrics'].append(metric)
            self.learning_rate *= 0.8

        return history
