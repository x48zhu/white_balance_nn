import math
import tensorflow as tf
from tensorflow.contrib.layers import flatten

mu = 0
# sigma = 1.0 / math.sqrt(NUM_FEATURES)
sigma = 0.02


def hyp_net_inference(input):
    # Common Convolutional Layers
    # Layer 1: Input = 47x47x2, Output = 10x10x128
    with tf.name_scope('Conv_1') as scope:
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 25, 3, 6), mean=mu, stddev=sigma), name="Weights")
        conv1_b = tf.Variable(tf.zeros(6), name="Bias")
        conv1 = tf.nn.conv2d(input, conv1_W, strides=(1, 1, 1, 1), padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Layer 2: Input = 10x10x128, Output = 4x4x256
    with tf.name_scope('Conv_2') as scope:
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name="Weights")
        conv2_b = tf.Variable(tf.zeros(16), name="Bias")
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=(1, 1, 1, 1), padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Flatten: Input = 4x4x256, Output = 4096
    fc0 = flatten(conv2)

    # Full Connected Layer
    with tf.name_scope('Fc_1') as scope:
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma), name="Weights")
        fc1_b = tf.Variable(tf.zeros(120), name="Bias")
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope('Fc_2') as scope:
        fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma), name="Weights")
        fc2_b = tf.Variable(tf.zeros(84), name="Bias")
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope('Fc_3') as scope:
        fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 2), mean=mu, stddev=sigma), name="Weights")
        fc3_b = tf.Variable(tf.zeros(2), name="Bias")
        output = tf.matmul(fc2, fc3_W) + fc3_b
        output = tf.nn.relu(output)

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

