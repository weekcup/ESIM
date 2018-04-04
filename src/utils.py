"""
Utility functions.
"""
# Aurelien Coet, 2018.

import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def prepare_data(preproc_dir, dataset, n_classes, max_length=None):
    """
    Load and prepare preprocessed data for the ESIM model.

    Args:
        preproc_dir: The path to the directory containing the preprocessed
                          data to be loaded.
        dataset: The type of the dataset that must be loaded (train, test or
                 dev).

    Returns:
        A tuple containing numpy arrays. The two first are the premises and
        hypotheses of the dataset padded with zeros to all have the same
        length. The third one is a numpy array containing the labels
        transformed to categorical form.
    """
    with open(os.path.join(preproc_dir, "premises_{}.pkl".format(dataset)),
              'rb') as premise_file:
        premises = pickle.load(premise_file)

    with open(os.path.join(preproc_dir, "hypotheses_{}.pkl".format(dataset)),
              'rb') as hypotheses_file:
        hypotheses = pickle.load(hypotheses_file)

    with open(os.path.join(preproc_dir, "labels_{}.pkl".format(dataset)),
              'rb') as labels_file:
        labels = pickle.load(labels_file)

    premises = pad_sequences(premises, maxlen=max_length,
                             padding='post', truncating='post')

    hypotheses = pad_sequences(hypotheses, maxlen=max_length,
                               padding='post', truncating='post')

    # Convert the labels to one-hot vectors.
    labels = to_categorical(labels, num_classes=n_classes)

    return (premises, hypotheses, labels)


def load_embeddings(filepath):
    """
    Load an embedding weights matrix from a pickle file.

    Args:
        filepath: The path to the file containing the embedding matrix.

    Returns:
        The embedding matrix.
    """
    with open(filepath, 'rb') as embed_file:
        embedding_weights = pickle.load(embed_file)

    return embedding_weights
