import tensorflow as tf
from model.module import Preprocess, Module1, Module2, Module3, Module4, Module5, InceptionAux
from collections import namedtuple

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])


class InceptionV3(tf.keras.Model):
    def __init__(self, num_class, aux_logits=True):
        super(InceptionV3, self).__init__()

        self.aux_logits = aux_logits
        self.preprocess = Preprocess()
        self.block_1 = tf.keras.Sequential([
            Module1(32),
            Module1(64),
            Module1(64),
        ])

        self.block_2 = tf.keras.Sequential([
            Module2(),
            Module3(128),
            Module3(160),
            Module3(160),
            Module3(192),
        ])

        if self.aux_logits:
            self.AuxLogits = InceptionAux(num_classes=num_class)

        self.block_3 = tf.keras.Sequential([
            Module4(),
            Module5(),
            Module5()
        ])

        self.avg_pool = tf.keras.layers.AvgPool2D((8, 8), strides=1, padding='same')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=num_class, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None, include_aux_logits=True):
        x = self.preprocess(inputs, training=training)
        x = self.block_1(x, training=training)
        x = self.block_2(x, training=training)

        if include_aux_logits and self.aux_logits:
            aux = self.AuxLogits(x)

        x = self.block_3(x, training=training)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        if include_aux_logits and self.aux_logits:
            return InceptionOutputs(x, aux)

        return x
