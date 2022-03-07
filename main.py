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

    args = parser.parse_args()
    data_dir = args.data

    train_x, train_y, test_x, test_y = load_data(data_dir)

    nn_model = model.NeuralNetworkModel(layers=3, learning_rate=1e0, activation='sigmoid', channels=128)
    history = nn_model.fit(train_x, train_y, epochs=10, shuffle=False, batch_size=2048)
    print('hi')
