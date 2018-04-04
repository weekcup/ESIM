"""
Preprocess the data necessary for the ESIM model.
"""
# Aurelien Coet, 2018.

import os
import sys
import numpy
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer


def read_data(filepath):
    """
    Read the premises, hypotheses and labels from a file in the SNLI dataset
    and return them in three separate lists.

    Args:
        filepath: The path to a file from the SNLI dataset.

    Returns:
        A dictionnary containing three lists, one for the premises, one for the
        hypotheses and one for the labels.
    """
    labels_dict = {"entailment": '0', "neutral": '1', "contradiction": '2'}
    filename = os.path.basename(filepath)

    premises = []
    hypotheses = []
    labels = []
    premises_lens = []
    hypotheses_lens = []

    with open(filepath, 'r') as input:
        # Ignore the first line containing headers.
        next(input)
        for line in input:
            line = line.strip().split('\t')
            if line[0] == '-':
                continue

            # Read the premise.
            sentence = line[5].rstrip()
            premises.append(sentence)
            premises_lens.append(len(sentence.split()))

            # Read the hypothesis.
            sentence = line[6].rstrip()
            hypotheses.append(sentence)
            hypotheses_lens.append(len(sentence.split()))

            # Read the label.
            labels.append(labels_dict[line[0]])

    print("Min. premise length: {}, max. premise length: {}"
          .format(min(premises_lens), max(premises_lens)))
    print("Min. hypothesis length: {}, max. hypothesis length: {}"
          .format(min(hypotheses_lens), max(hypotheses_lens)))

    return {"premises": premises, "hypotheses": hypotheses,
            "labels": labels}


def save_preprocessed_data(tokenizer, data, dataset, targetdir):
    """
    Save the preprocessed data to pickle files for later use. The preprocessed
    data consists in the premises and hypotheses with their words transformed
    to their indices, and the labels transformed to integer values.

    Args:
        tokenizer: A Keras Tokenizer object that has already been fit on
                   the training data (its word_index has been built).
        data: A dictionnary containing lists of strings for the sentences in
              the premises and hypotheses, as well as a list with their
              associated labels.
        dataset: A string indicating the type of dataset being saved (train,
                 test or dev).
        targetdir: The target directory in which to save the pickled files.
    """
    # Transform the words in the input data to their indexes and save them
    # in separate pickle files for the premises and hypotheses.
    with open(os.path.join(targetdir, "premises_{}.pkl".format(dataset)),
              'wb') as output:
        pickle.dump(tokenizer.texts_to_sequences(data["premises"]), output)

    with open(os.path.join(targetdir, "hypotheses_{}.pkl".format(dataset)),
              'wb') as output:
        pickle.dump(tokenizer.texts_to_sequences(data["hypotheses"]), output)

    # Pickle the labels too.
    with open(os.path.join(targetdir, "labels_{}.pkl".format(dataset)),
              'wb') as output:
        pickle.dump(data["labels"], output)


def build_embedding_weights(worddict, embeddings_file, targetdir):
    """
    Build the embedding weights matrix from a words dictionnary and existing
    embeddings, and save it in pickled form.

    Args:
        worddict: A dictionnary of words with their associated integer index.
        embeddings_file: A file containing predefined word embeddings.
        targetdir: The path to the target directory where to save the embedding
                   weights matrix.
    """
    print("* Loading word embeddings from {}...".format(embeddings_file))
    # Load the word embeddings in a dictionnary.
    with open(embeddings_file, 'r') as input:
        embeddings = {}
        for line in input:
            line = line.split()
            # Ignore lines corresponding to words separated by spaces in the
            # predefined embeddings.
            if len(line[1:]) != 300:
                continue
            word = line[0]
            if word in worddict:
                last = word
                embeddings[word] = line[1:]

    print("* Building embedding weights matrix...")
    # Initialize the embedding weights matrix.
    num_words = len(worddict)
    dims = len(embeddings[last])
    embedding_weights = np.zeros((num_words, dims))

    # Build the embedding weights matrix.
    for word, i in worddict.items():
        if word in embeddings:
            embedding_weights[i] = embeddings[word]
        else:
            # Out of vocabulary words are initialised with random gaussian
            # samples.
            embedding_weights[i] = np.random.normal(size=(dims))

    # Save the matrix in pickled form.
    with open(os.path.join(targetdir, "embedding_weights.pkl"),
              'wb') as output:
        pickle.dump(embedding_weights, output)


def preprocess_data(train_file, test_file, dev_file, embeddings_file,
                    targetdir):
    """
    Preprocess the data for the ESIM model. Compute the word indices from the
    training data, transform all words in the train/test/dev datasets to their
    indices, save them in pickled files, and finally build the embedding matrix
    and save it in pickled form.

    Args:
        train_file: The path to the file containing the training data from the
                    SNLI dataset.
        test_file: The path to the file containing the test data from the SNLI
                   dataset.
        dev_file: The path to the file containing the dev data from the SNLI
                  dataset.
        embeddings_file: The path to the file containing the word embeddings to
                         use for the embedding matrix.
        targetdir: The path to the target directory for the pickled files
                   produced by the function.
    """
    print(20*"=" + "Processing train data..." + 20*"=")
    data = read_data(train_file)

    # Build the dictionnary of words from the training data with Keras'
    # Tokenizer class. A special token is created for out of voc. words
    # (token '__OOV__'), and index 0 is reserved for padding.
    print("* Building word index dictionnary...")
    tokenizer = Tokenizer(lower=False, oov_token="__OOV__")
    tokenizer.fit_on_texts(data["premises"]+data["hypotheses"])
    tokenizer.word_index["__PAD__"] = 0
    print("** Total number of words: {}".format(len(tokenizer.word_index)))
    # Save the dictionnary in a pickle file.
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    with open(os.path.join(targetdir, "worddict.pkl"), 'wb') as pkl_f:
        pickle.dump(tokenizer.word_index, pkl_f)

    print("* Transforming and saving train data...")
    save_preprocessed_data(tokenizer, data, "train", targetdir)

    # Preprocess and save the test dataset.
    print(20*"=" + "Processing test data..." + 20*"=")
    data = read_data(test_file)
    print("* Transforming and saving test data...")
    save_preprocessed_data(tokenizer, data, "test", targetdir)

    # Preprocess and save the dev dataset.
    print(20*"=" + "Processing dev data..." + 20*"=")
    data = read_data(dev_file)
    print("* Transforming and saving dev data...")
    save_preprocessed_data(tokenizer, data, "dev", targetdir)

    # Create and save the embedding weights matrix.
    print(20*"=" + "Building embedding weights matrix..." + 20*"=")
    build_embedding_weights(tokenizer.word_index, embeddings_file, targetdir)


if __name__ == "__main__":
    basedir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "..", "data")
    snli_dir = os.path.join(basedir, "snli", "snli_1.0")
    glove_dir = os.path.join(basedir, "glove")
    targetdir = os.path.join(basedir, "preprocessed")

    preprocess_data(os.path.join(snli_dir, "snli_1.0_train.txt"),
                    os.path.join(snli_dir, "snli_1.0_test.txt"),
                    os.path.join(snli_dir, "snli_1.0_dev.txt"),
                    os.path.join(glove_dir, "glove.840B.300d.txt"),
                    targetdir)
