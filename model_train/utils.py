import numpy as np
import tensorflow as tf


def conv_block(inputs=None, filter=None, kernel=None, strides=None, padding='same'):
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(filter, kernel, strides, padding=padding)(x)
    return x


def residual_block(inputs=None, filter=None, kernel=3, strides=None, padding='same'):
    x = tf.keras.layers.Conv1D(filter, kernel, strides, padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights


def self_attention(pos_encoding, dropout):
    x, attention_weights = scaled_dot_product_attention(pos_encoding, pos_encoding, pos_encoding)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    outputs = tf.keras.layers.Add()([pos_encoding, x])
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

    return outputs


def fc_block(inputs=None, units=None, dropout=None):
    x = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x
