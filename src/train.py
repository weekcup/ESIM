"""
Train the ESIM model on some dataset.
"""

import os
import argparse
from keras.callbacks import ModelCheckpoint
from model import ESIM
from utils import prepare_data, load_embeddings


def train(preproc_dir, n_classes, max_length, hidden_units, dropout,
          batch_size, epochs, output_dir):
    """
    """
    print("Loading training and validation data...")
    train_premises, train_hyps, train_labels = prepare_data(preproc_dir,
                                                            'train',
                                                            n_classes,
                                                            max_length)
    valid_premises, valid_hyps, valid_labels = prepare_data(preproc_dir,
                                                            'dev',
                                                            n_classes,
                                                            max_length)

    print("Loading embedding weights...")
    embedding_weights = load_embeddings(os.path.join(preproc_dir,
                                                     "embedding_weights.pkl"))

    # Build the model.
    esim = ESIM(n_classes, embedding_weights, max_length, hidden_units,
                dropout)
    model = esim.build_model()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir,
                            "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    model.fit(x=[train_premises, train_hyps],
              y=train_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([valid_premises, valid_hyps], valid_labels),
              callbacks=[checkpoint],
              shuffle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the ESIM model')
    parser.add_argument('preproc_dir', help='Path to the directory containing\
 the preprocessed data to be used to train the model.')
    parser.add_argument('output_dir', help='Path to the directory where the\
 learned weights of the model must be saved.')
    parser.add_argument('--epochs', type=int, default=64, help='Number of\
 epochs to run for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of the\
 mini-batches to use during training.')
    parser.add_argument('--hidden_units', type=int, default=300, help='Number\
 of hidden units to use in the layers of the model')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout\
 rate to use during training.')
    parser.add_argument('--nclasses', type=int, default=3, help='Number of\
 classes.')
    parser.add_argument('--max_length', type=int, default=100, help='Max.\
 length of the sentences for the premise and hypothesis.')

    args = parser.parse_args()

    print("Starting training of the model...")
    train(args.preproc_dir, args.nclasses, args.max_length, args.hidden_units,
          args.dropout, args.batch_size, args.epochs, args.output_dir)
