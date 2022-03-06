import argparse
import gzip
import numpy as np
import os
import requests
import shutil

from tqdm import tqdm

LABEL_MAGIC_NUM = 2049
IMAGE_MAGIC_NUM = 2051

train_image_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_label_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
test_image_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_label_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
urls = [train_image_url, train_label_url, test_image_url, test_label_url]


def download_data(data):
    os.makedirs(os.path.join(data, 'raw'), exist_ok=True)
    for url in urls:
        request = requests.get(url)
        with open(os.path.join(data, 'raw', url.split('/')[-1]), 'wb') as file:
            file.write(request.content)


def extract_data(data):
    os.makedirs(os.path.join(data, 'extracted'), exist_ok=True)
    for url in urls:
        url_split = url.split('/')[-1]
        with gzip.open(os.path.join(data, 'raw', url_split), 'rb') as f_in:
            with open(os.path.join(data, 'extracted', url_split), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_labels(path):
    labels = []
    with open(path, 'rb') as label_file:
        assert int.from_bytes(label_file.read(4), 'big') == LABEL_MAGIC_NUM
        num_labels = int.from_bytes(label_file.read(4), 'big')
        for _ in tqdm(range(num_labels)):
            labels.append(int.from_bytes(label_file.read(1), 'big'))
    return np.array(labels, dtype=np.uint8)


def load_images(path):
    images = []
    with open(path, 'rb') as label_file:
        assert int.from_bytes(label_file.read(4), 'big') == IMAGE_MAGIC_NUM
        num_images = int.from_bytes(label_file.read(4), 'big')
        num_rows = int.from_bytes(label_file.read(4), 'big')
        num_cols = int.from_bytes(label_file.read(4), 'big')
        for _ in tqdm(range(num_images)):
            image_pixels = []
            for row in range(num_rows):
                row_pixels = []
                for col in range(num_cols):
                    row_pixels.append(int.from_bytes(label_file.read(1), 'big'))
                image_pixels.append(row_pixels)
            images.append(image_pixels)
    return np.array(images, dtype=np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='data', help='Directory for the data to be stored.')

    args = parser.parse_args()
    data_dir = args.data

    os.makedirs(data_dir, exist_ok=True)

    print('Downloading data...')
    download_data(data_dir)
    print('Data downloaded!')

    print('Extracting data...')
    extract_data(data_dir)
    print('Data extracted!')

    print('Loading data...')
    train_labels = load_labels(os.path.join(data_dir, 'extracted', train_label_url.split('/')[-1]))
    train_images = load_images(os.path.join(data_dir, 'extracted', train_image_url.split('/')[-1]))
    test_labels = load_labels(os.path.join(data_dir, 'extracted', test_label_url.split('/')[-1]))
    test_images = load_images(os.path.join(data_dir, 'extracted', test_image_url.split('/')[-1]))
    print('Data loaded!')

    print('Saving data...')
    os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
    np.save(os.path.join(data_dir, 'processed', 'train_labels.npy'), train_labels)
    np.save(os.path.join(data_dir, 'processed', 'train_images.npy'), train_images)
    np.save(os.path.join(data_dir, 'processed', 'test_labels.npy'), test_labels)
    np.save(os.path.join(data_dir, 'processed', 'test_images.npy'), test_images)
    print('Data saved!')
