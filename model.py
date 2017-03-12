import math
import tensorflow as tf
from tensorflow.contrib.layers import flatten

mu = 0
sigma = 0.1


def hyp_net_inference(input):
    # Common Convolutional Layers
    # Layer 1: Input = 47x47x2, Output = 10x10x128
    with tf.name_scope('Hyp_Conv_1') as scope:
        conv1_W = tf.Variable(tf.truncated_normal(shape=(8, 8, 3, 128), mean=mu, stddev=sigma), name="Weights")
        conv1_b = tf.Variable(tf.zeros(128), name="Bias")
        conv1 = tf.nn.conv2d(input, conv1_W, strides=(1, 4, 4, 1), padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)
        # conv1 = tf.nn.max_pool(conv1, ksize=[], strides=[], padding='VALID')

    # Layer 2: Input = 10x10x128, Output = 4x4x256
    with tf.name_scope('Hyp_Conv_2') as scope:
        conv2_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 128, 256), mean=mu, stddev=sigma), name="Weights")
        conv2_b = tf.Variable(tf.zeros(256), name="Bias")
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=(1, 2, 2, 1), padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        # conv2 = tf.nn.max_pool(conv2, ksize=[], strides=[], padding='VALID')

    # Flatten: Input = 4x4x256, Output = 4096
    fc0 = flatten(conv2)

    # Branch A Full Connected Layer
    with tf.name_scope('Hyp_fc_A_1') as scope:
        fc1A_W = tf.Variable(tf.truncated_normal(shape=(4096, 256), mean=mu, stddev=sigma), name="Weights")
        fc1A_b = tf.Variable(tf.zeros(256), name="Bias")
        fc1A = tf.matmul(fc0, fc1A_W) + fc1A_b
        fc1A = tf.nn.relu(fc1A)

    with tf.name_scope('Hyp_fc_A_2') as scope:
        fc2A_W = tf.Variable(tf.truncated_normal(shape=(256, 2), mean=mu, stddev=sigma), name="Weights")
        fc2A_b = tf.Variable(tf.zeros(2), name="Bias")
        outputA = tf.matmul(fc1A, fc2A_W) + fc2A_b
        outputA = tf.nn.relu(outputA)

    # Branch B Full Connected Layer
    with tf.name_scope('Hyp_fc_B_1') as scope:
        fc1B_W = tf.Variable(tf.truncated_normal(shape=(4096, 256), mean=mu, stddev=sigma), name="Weights")
        fc1B_b = tf.Variable(tf.zeros(256), name="Bias")
        fc1B = tf.matmul(fc0, fc1B_W) + fc1B_b
        fc1B = tf.nn.relu(fc1B)

    with tf.name_scope('Hyp_fc_B_2') as scope:
        fc2B_W = tf.Variable(tf.truncated_normal(shape=(256, 2), mean=mu, stddev=sigma), name="Weights")
        fc2B_b = tf.Variable(tf.zeros(2), name="Bias")
        outputB = tf.matmul(fc1B, fc2B_W) + fc2B_b
        outputB = tf.nn.relu(outputB)

    return outputA, outputB


# Hypothesis network uses Euclidean loss
# Also note that because of "winner-take-all" learning scheme, (only the better 
#   one of the branches is optimized) the loss is only reflect the branch with smaller 
#   the Euclidean loss.
def hyp_net_loss(outputA, outputB, labels):
    with tf.name_scope('Hyp_Loss') as scope:
        errorA = tf.square(tf.subtract(outputA, labels), name="Error_A")
        errorB = tf.square(tf.subtract(outputB, labels), name="Error_B")
        min_error = tf.select(tf.less(errorA, errorB), errorA, errorB, name="Min_Error")
        loss = tf.reduce_mean(min_error, name="Loss")
    return loss


def hyp_net_training(loss, learning_rate):
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op


def hyp_net_evaluation(outputA, outputB, labels):
    with tf.name_scope('Hyp_Eval') as scope:
        errorA = tf.square(tf.subtract(outputA, labels), name="Error_A")
        errorB = tf.square(tf.subtract(outputB, labels), name="Error_B")
        min_error = tf.select(tf.less(errorA, errorB), errorA, errorB, name="Min_Error")
        loss = tf.reduce_mean(min_error, name="Loss")
    return loss


# Then the score (sA,sB) for the branch whose hypothesis is closer to the ground
#   truth is set to 1 and the other one to 0. 
def calc_ground_truth_score(outputA, outputB, labels):
    with tf.name_scope('Hyp_Score') as scope:
        errorA = tf.reduce_sum(tf.square(tf.subtract(outputA, labels)), 1)
        errorB = tf.reduce_sum(tf.square(tf.subtract(outputB, labels)), 1)
        zeros = tf.zeros_like(errorA)
        ones = tf.ones_like(errorA)
        chooseA = tf.pack([ones, zeros], axis=1)
        chooseB = tf.pack([zeros, ones], axis=1)
        ground_truth_score = tf.select(tf.less(errorA, errorB), chooseA, chooseB, name="One_Or_Zero")
    return ground_truth_score


#####################################################################################
# Selection Network

def sel_net_inference(input):
    # Common Convolutional Layers
    with tf.name_scope('Sel_Conv_1') as scope:
        # Layer 1: Input = 47x47x2, Output = 10x10x128
        conv1_W = tf.Variable(tf.truncated_normal(shape=(8, 8, 3, 128), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(128))
        conv1 = tf.nn.conv2d(input, conv1_W, strides=(1, 4, 4, 1), padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)
        # conv1 = tf.nn.max_pool(conv1, ksize=[], strides=[], padding='VALID')

    # Layer 2: Input = , Output = 4x4x256
    with tf.name_scope('Sel_Conv_2') as scope:
        conv2_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 128, 256), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(256))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=(1, 2, 2, 1), padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        # conv2 = tf.nn.max_pool(conv2, ksize=[], strides=[], padding='VALID')

    # Flatten: Input = 4x4x256, Output = 4096
    fc0 = flatten(conv2)
    with tf.name_scope('Sel_fc_1') as scope:
        fc1_W = tf.Variable(tf.truncated_normal(shape=(4096, 256), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(256))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope('Sel_fc_2') as scope:
        fc2_W = tf.Variable(tf.truncated_normal(shape=(256, 2), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(2))
        logits = tf.matmul(fc1, fc2_W) + fc2_b

    return logits


# Selection network uses multinomial logistic loss
def sel_net_loss(logits, labels):
    with tf.name_scope('Sel_Loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    return tf.reduce_mean(cross_entropy)


def sel_net_training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op


def sel_net_evaluation(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    return tf.reduce_mean(cross_entropy)


#####################################################################################
#
########################### least_square_error ######################################

def least_square_error(output, labels):
    with tf.name_scope('Least_square_error') as scope:
        loss = tf.reduce_mean(tf.square(tf.sub(output, labels), name="Error"))
    return loss


########################## angular_error ############################################

def num(output, labels):
    numerator = tf.reduce_sum(tf.mul(output, labels), axis=1)
    return numerator


def denom(output, labels):
    denominator = tf.mul(tf.sqrt(tf.reduce_sum(tf.mul(output, output), axis=1)), \
                         tf.sqrt(tf.reduce_sum(tf.mul(labels, labels), axis=1)))
    return denominator


def angular_error(numerator, denominator):
    with tf.name_scope('Angular_error') as scope:
        loss = tf.reduce_mean(tf.acos(tf.div(numerator, denominator)))
    return loss


#####################################################################################

def inference(outputA, outputB, output_sel):
    """
    Use estimation from A, B hyp-net and score from sel-net to decide the
    final estimation of the patches.
    Args:
        outputA: illuminant estimation from batch A, dimension nx2
        outputB: illuminant estimation from batch B, dimension nx2
        output_sel: score on outputA and outputB from Selection Network, dimension nx2

    Returns:
        The final illuminant estimation for the patches, dimension nx2
    """
    with tf.name_scope("Inference") as scope:
        output = tf.select(output_sel[:, 0] > output_sel[:, 1], outputA, outputB)
    return output


def evaluation(output, labels):
    with tf.name_scope("Evaluation") as scope:
        # loss = least_square_error(output, labels)
        loss = angular_error(output, labels)
    return loss
