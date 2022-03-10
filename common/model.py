import numpy as np
from common import activations
from common import metrics
from math import ceil
from tqdm import tqdm

# RNG with seed for all randomization of weights, biases, and batches.
rng = np.random.default_rng(12345)


class NeuralNetworkModel:
    def __init__(self, in_features=784, out_features=10, layers=3, channels=16, activation='sigmoid'):
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = layers
        self.channels = channels
        self.activation = activation

        # Initialize learning rate here, but will be set in self.fit()
        self.learning_rate = 1e-3

        self.layers = []
        for layer in range(layers):
            in_f = in_features if layer == 0 else channels
            out_f = out_features if layer == layers - 1 else channels
            self.layers.append(rng.uniform(-0.2, 0.2, size=[in_f + 1, out_f]))

    # Predict a single forward pass of batch. Returns the output of the last layer and the layer_logits containing the
    #   predictions for every layer.
    # layer_logits: list of length self.layers+1. Each item in the list is a sublist of [xz, xa], or the outputs
    #   before and after activation for that layer, respectively. Used for backpropagation.
    def forward_pass(self, inputs):
        xa = inputs
        layer_logits = [[0., xa]]
        for i in range(len(self.layers)):
            xz = np.matmul(xa, self.layers[i][:-1]) + self.layers[i][-1]
            xa = activations.activation_dict[self.activation][0](xz)
            layer_logits.append([xz, xa])
        return xa, layer_logits

    # Calculate gradient vector for a series of logits and labels.
    def _get_gradient(self, logits, labels):
        a = logits[-1][0]
        true = labels
        d_a = metrics.d_mse(true, a)
        gradient_vector = []
        for i in range(-1, -len(self.layers) - 1, -1):
            layer = self.layers[i]
            a = logits[i-1][1]
            z = logits[i][0]
            d_z = activations.activation_dict[self.activation][1](z)
            d_w = (d_a * d_z)[:, np.newaxis, :] * a[:, :, np.newaxis]
            d_b = d_a * d_z
            gradient_vector.insert(0, np.concatenate([d_w, d_b[:, np.newaxis, :]], axis=-2).mean(axis=0))
            d_a = np.matmul(layer[:-1], (d_a * d_z)[:, :, np.newaxis])[:, :, 0]
        return gradient_vector

    # Optimizer step. Compute gradient vector, and then adjust the weights and biases according to it. Standard
    #   stochastic gradient descent.
    def _optimizer_step(self, logits, labels):
        gv = self._get_gradient(logits, labels)
        for layer, gradient in zip(self.layers, gv):
            layer -= gradient * self.learning_rate

    # Train step. Forward pass and optimizer step.
    def _train_step(self, inputs, labels):
        y, logits = self.forward_pass(inputs)
        self._optimizer_step(logits, labels)
        return y

    # Test step. Forward pass only
    def _test_step(self, inputs):
        y, _ = self.forward_pass(inputs)
        return y

    # Fit model to input data. Similar in use to TensorFlow with Keras's model.fit(). Returns a history dict containing
    #   the losses and metrics for both training and validation for every epoch.
    def fit(self, inputs, labels, test_inputs, test_labels, epochs=1, batch_size=64, shuffle=True, learning_rate=6e-1):
        train_x = inputs
        train_y = labels
        test_x = test_inputs
        test_y = test_labels

        self.learning_rate = learning_rate

        history = {'losses': [], 'metrics': [], 'val_losses': [], 'val_metrics': []}

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
                batch_y_pred = self._train_step(batch_x, batch_y)
                loss.append(metrics.mse(batch_y, batch_y_pred))
                metric.append(metrics.categorical_acc(batch_y, batch_y_pred))

            loss = np.concatenate(loss).mean()
            metric = np.array(metric).sum() / len(inputs)

            print(f'Loss: {loss}, Metric: {metric}', end='; ')

            val_loss = []
            val_metric = []
            for batch in range(ceil(len(test_inputs) / batch_size)):
                batch_x = test_x[batch*batch_size:(batch+1)*batch_size]
                batch_y = test_y[batch*batch_size:(batch+1)*batch_size]
                batch_y_pred = self._test_step(batch_x)
                val_loss.append(metrics.mse(batch_y, batch_y_pred))
                val_metric.append(metrics.categorical_acc(batch_y, batch_y_pred))

            val_loss = np.concatenate(val_loss).mean()
            val_metric = np.array(val_metric).sum() / len(test_inputs)

            print(f'Val Loss: {val_loss}, Metric: {val_metric}\n')

            history['losses'].append(loss)
            history['metrics'].append(metric)
            history['val_losses'].append(val_loss)
            history['val_metrics'].append(val_metric)

        return history
