import argparse
import numpy as np
import os

from common import model


def convert_labels(labels):
    labels_converted = []
    for label in labels:
        lc = np.zeros(10)
        lc[label] = 1.
        labels_converted.append(lc)
    return np.stack(labels_converted)


def load_data(data):
    train_l = np.load(os.path.join(data, 'processed', 'train_labels.npy'))
    test_l = np.load(os.path.join(data, 'processed', 'test_labels.npy'))
    train_l = convert_labels(train_l).astype(float)
    test_l = convert_labels(test_l).astype(float)

    train_i = np.load(os.path.join(data, 'processed', 'train_images.npy')).astype(float)
    test_i = np.load(os.path.join(data, 'processed', 'test_images.npy')).astype(float)
    train_i = np.reshape(train_i, [-1, 28 * 28]) / 255.
    test_i = np.reshape(test_i, [-1, 28 * 28]) / 255.
    return train_i, train_l, test_i, test_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='data', help='Directory for the data to be loaded from.')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers in model, not including input layer')
    parser.add_argument('--channels', type=int, default=16, help='Number of channels in hidden layers')
    parser.add_argument('--in-features', type=int, default=784, help='Number of pixels per image. Should stay as 784')
    parser.add_argument('--out-features', type=int, default=10, help='Number of outputs to predict. Should stay as 10')
    parser.add_argument('--activation', default='sigmoid', choices=['sigmoid', 'relu', 'linear'],
                        help='Activation to use after each layer')
    parser.add_argument('--learning-rate', type=float, default=6e-1, help='Learning rate for SGD')
    parser.add_argument('--batch', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')

    args = parser.parse_args()
    data_dir = args.data
    layers = args.layers
    channels = args.channels
    in_features = args.in_features
    out_features = args.out_features
    activation = args.activation
    learning_rate = args.learning_rate
    batch = args.batch
    epochs = args.epochs

    train_x, train_y, test_x, test_y = load_data(data_dir)

    nn_model = model.NeuralNetworkModel(in_features=in_features, out_features=out_features, layers=layers,
                                        channels=channels, activation=activation)
    history = nn_model.fit(train_x, train_y, test_x, test_y, epochs=epochs, shuffle=True, batch_size=batch,
                           learning_rate=learning_rate)

    print(f"Final loss: {history['losses'][-1]}, accuracy: {history['metrics'][-1]*100.:.2f}%, ", end='')
    print(f"val_loss: {history['val_losses'][-1]}, val_accuracy: {history['val_metrics'][-1]*100.:.2f}%")
