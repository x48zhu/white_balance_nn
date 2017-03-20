import math
import tensorflow as tf
from tensorflow.contrib.layers import flatten

mu = 0
# sigma = 1.0 / math.sqrt(NUM_FEATURES)
sigma = 0.01

def full_connect_layer(input_layer, name='Fc'):
    with tf.name_scope(name + '_1') as scope:
        fc1A_W = tf.Variable(tf.truncated_normal(shape=(4096, 256), mean=mu, stddev=sigma), name="Weights")
        fc1A_b = tf.Variable(tf.zeros(256), name="Bias")
        fc1A = tf.matmul(input_layer, fc1A_W) + fc1A_b
        fc1A = tf.nn.relu(fc1A)

    with tf.name_scope(name + '_2') as scope:
        fc2A_W = tf.Variable(tf.truncated_normal(shape=(256, 2), mean=mu, stddev=sigma), name="Weights")
        fc2A_b = tf.Variable(tf.zeros(2), name="Bias")
        output = tf.matmul(fc1A, fc2A_W) + fc2A_b
        output = tf.nn.relu(output)

    return output


def hyp_net_inference(input):
    # Common Convolutional Layers
    # Layer 1: Input = 47x47x2, Output = 10x10x128
    with tf.name_scope('Conv_1') as scope:
        conv1_W = tf.Variable(tf.truncated_normal(shape=(8, 8, 3, 128), mean=mu, stddev=sigma), name="Weights")
        conv1_b = tf.Variable(tf.zeros(128), name="Bias")
        conv1 = tf.nn.conv2d(input, conv1_W, strides=(1, 4, 4, 1), padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)
        # conv1 = tf.nn.max_pool(conv1, ksize=[], strides=[], padding='VALID')

    # Layer 2: Input = 10x10x128, Output = 4x4x256
    with tf.name_scope('Conv_2') as scope:
        conv2_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 128, 256), mean=mu, stddev=sigma), name="Weights")
        conv2_b = tf.Variable(tf.zeros(256), name="Bias")
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=(1, 2, 2, 1), padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        # conv2 = tf.nn.max_pool(conv2, ksize=[], strides=[], padding='VALID')

    # Flatten: Input = 4x4x256, Output = 4096
    fc0 = flatten(conv2)

    # Full Connected Layer
    output = full_connect_layer(fc0)

    return output


# Hypothesis network uses Euclidean loss
def hyp_net_loss(output, labels):
    with tf.name_scope('Hyp_Loss') as scope:
        loss = tf.reduce_mean(tf.square(tf.subtract(output, labels), name="Square_loss"))
    return loss


def hyp_net_training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

