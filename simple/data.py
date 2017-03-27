import csv
import numpy as np
from sklearn.model_selection import train_test_split
import os

from simple.constants import data_dir
from utils import logger


def load_data(data_set_name, valid_percent=0.2, test_percent=0.2, debug=False):
    features = []
    labels = []
    # Load features and labels
    csv_file = open(os.path.join(data_dir, "Cheng-Prasad-Brown", data_set_name + '.csv'), 'r')
    # csv_file = open(os.path.join(data_dir, data_set_name + '.csv'), 'r')
    csv_file_object = csv.reader(csv_file)

    for row in csv_file_object:
        row = [float(i) for i in row]
        features.append(row[:8])
        labels.append(row[-2:])
    assert len(features) == len(labels)
    logger.info("Data loaded. Total data size: %d" % len(features))
    if debug:
        logger.debug("%s %s" % (str(features[0]), str(labels[0])))

    return DataSets(np.array(features),
                    np.array(labels),
                    valid_percent,
                    test_percent)


class DataSet(object):
    def __init__(self, features, labels, normalize=True):
        assert features.shape[0] == labels.shape[0]
        self._num_examples = features.shape[0]

        if normalize:
            # Convert from [0, 255] -> [0.0, 1.0].
            pass
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set.

        Args:
            batch_size: The size of batch to retrieve
        """

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]


class DataSets(object):
    def __init__(self, features, labels, valid_percent, test_percent):
        train_features, test_features, train_labels, test_labels = \
            train_test_split(features,
                             labels,
                             test_size=test_percent,
                             random_state=0)
        train_features, val_features, train_labels, val_labels = \
            train_test_split(train_features,
                             train_labels,
                             test_size=valid_percent,
                             random_state=0)
        print(val_features.shape)
        self.train = DataSet(np.array(train_features), np.array(train_labels))
        self.valid = DataSet(np.array(val_features), np.array(val_labels))
        self.test = DataSet(np.array(test_features), np.array(test_labels))