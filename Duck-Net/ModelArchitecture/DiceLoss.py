import keras.backend as K
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy


def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = K.cast(ground_truth, tf.float32)
    predictions = K.cast(predictions, tf.float32)
    ground_truth = K.flatten(ground_truth)
    predictions = K.flatten(predictions)
    intersection = K.sum(predictions * ground_truth)
    union = K.sum(predictions) + K.sum(ground_truth)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1 - dice


def classification_loss(ground_truth, predictions):
    return tf.keras.losses.sparse_categorical_crossentropy(ground_truth, predictions)
