"""
Definition of the ESIM model.

Inspired from the code on:
https://github.com/yuhsinliu1993/Quora_QuestionPairs_DL
"""

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from layers import *


class ESIM(object):
    """
    ESIM model for Natural Language Inference (NLI) tasks.
    """

    def __init__(self, n_classes, embedding_weights, max_length, hidden_units,
                 dropout=0.5, learning_rate=0.0004):
        self.n_classes = n_classes
        self.embedding_weights = embedding_weights
        self.voc_size, self.embedding_dim = embedding_weights.shape
        self.max_length = max_length
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate

    def build_model(self):
        """
        Build the model.

        Returns:
            The ESIM model compiled and ready to be trained.
        """
        a = Input(shape=(self.max_length,), dtype='int32', name='premise')
        b = Input(shape=(self.max_length,), dtype='int32', name='hypothesis')

        # ---------- Embedding layer ---------- #
        embedding = EmbeddingLayer(self.voc_size, self.embedding_dim,
                                   self.embedding_weights,
                                   max_length=self.max_length)

        embedded_a = embedding(a)
        embedded_b = embedding(b)

        # ---------- Encoding layer ---------- #
        encoded_a = EncodingLayer(self.hidden_units,
                                  self.max_length,
                                  dropout=self.dropout)(embedded_a)
        encoded_b = EncodingLayer(self.hidden_units,
                                  self.max_length,
                                  dropout=self.dropout)(embedded_b)

        # ---------- Local inference layer ---------- #
        m_a, m_b = LocalInferenceLayer()([encoded_a, encoded_b])

        # ---------- Inference composition layer ---------- #
        composed_a = InferenceCompositionLayer(self.hidden_units,
                                               self.max_length,
                                               dropout=self.dropout)(m_a)
        composed_b = InferenceCompositionLayer(self.hidden_units,
                                               self.max_length,
                                               dropout=self.dropout)(m_b)

        # ---------- Pooling layer ---------- #
        pooled = PoolingLayer()([composed_a, composed_b])

        # ---------- Classification layer ---------- #
        prediction = MLPLayer(self.hidden_units, self.n_classes,
                              dropout=self.dropout)(pooled)

        model = Model(inputs=[a, b], outputs=prediction)
        model.compile(optimizer=Adam(lr=self.learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model
