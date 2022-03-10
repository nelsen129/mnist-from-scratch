import argparse
import numpy as np
import os
from common import model
from matplotlib import pyplot as plt


def _convert_labels(labels):
    labels_converted = []
    for label in labels:
        lc = np.zeros(10)
        lc[label] = 1.
        labels_converted.append(lc)
    return np.stack(labels_converted)


def load_data(data):
    try:
        train_l = np.load(os.path.join(data, 'processed', 'train_labels.npy'))
        test_l = np.load(os.path.join(data, 'processed', 'test_labels.npy'))
        train_l = _convert_labels(train_l).astype(float)
        test_l = _convert_labels(test_l).astype(float)

        train_i = np.load(os.path.join(data, 'processed', 'train_images.npy')).astype(float)
        test_i = np.load(os.path.join(data, 'processed', 'test_images.npy')).astype(float)
        train_i = np.reshape(train_i, [-1, 28 * 28]) / 255.
        test_i = np.reshape(test_i, [-1, 28 * 28]) / 255.
    except FileNotFoundError:
        return None, None, None, None
    return train_i, train_l, test_i, test_l


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='data', help='Directory for the data to be loaded from.')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers in model, not including input layer')
    parser.add_argument('--channels', type=int, default=128, help='Number of channels in hidden layers')
    parser.add_argument('--in-features', type=int, default=784, help='Number of pixels per image. Should stay as 784')
    parser.add_argument('--out-features', type=int, default=10, help='Number of outputs to predict. Should stay as 10')
    parser.add_argument('--activation', default='sigmoid', choices=['sigmoid', 'relu', 'linear'],
                        help='Activation to use after each layer')
    parser.add_argument('--learning-rate', type=float, default=1.4e0, help='Learning rate for SGD')
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

    if not os.path.isdir(data_dir):
        print(f'Error: data path at {data_dir} does not exist. Please run process_data.py to create it.')
        return 1

    train_x, train_y, test_x, test_y = load_data(data_dir)

    if train_x is None:
        print(f'Error: data loading did not work from {data_dir}. Please run process_data.py to fix it.')
        return 1

    # Create model
    nn_model = model.NeuralNetworkModel(in_features=in_features, out_features=out_features, layers=layers,
                                        channels=channels, activation=activation)

    # Fit model
    history = nn_model.fit(train_x, train_y, test_x, test_y, epochs=epochs, shuffle=True, batch_size=batch,
                           learning_rate=learning_rate)

    print(f"Final loss: {history['losses'][-1]}, accuracy: {history['metrics'][-1]*100.:.2f}%, ", end='')
    print(f"val_loss: {history['val_losses'][-1]}, val_accuracy: {history['val_metrics'][-1]*100.:.2f}%")

    fig, axs = plt.subplots(2)
    fig.suptitle("Training History")
    axs[0].plot(range(epochs), history['losses'], 'b-', range(epochs), history['val_losses'], 'r-')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_yscale("log")
    axs[0].legend(["Train", "Val"])
    axs[0].set_title("Loss Per Epoch")
    axs[1].plot(range(epochs), history['metrics'], 'b-', range(epochs), history['val_metrics'], 'r-')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_yticks([i for i in np.arange(0.75, 1.00, 0.05)], [f'{i*100.:.1f}%' for i in np.arange(0.75, 1.00, 0.05)])
    axs[1].legend(["Train", "Val"])
    axs[1].set_title("Accuracy Per Epoch")
    plt.show()


if __name__ == '__main__':
    main()

# 1.8e0, 16*2, 93.71%