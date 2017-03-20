import math
import tensorflow as tf

from utils import angular_error_scalar

NUM_CLASSES = 2
NUM_FEATURES = 8


def inference(images, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.
    Args:
      images: Images placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.
    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([NUM_FEATURES, hidden1_units],
                                stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(output, labels):
    """Calculates the loss from the logits and the labels.
    Args:
      output: output tensor, float - [batch_size, 2].
      labels: Labels tensor, int32 - [batch_size, 2].
    Returns:
      loss: Loss tensor of type float.
    """
    square_loss = tf.square(tf.subtract(output, labels), name="Square_loss")
    return tf.reduce_mean(square_loss, name='Square_loss_mean')


def training(sqaure_mean_loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      sqaure_mean_loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('Loss', sqaure_mean_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(sqaure_mean_loss, global_step=global_step)
    return train_op


def evaluation(output, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      output: output tensor, float - [batch_size, 2].
      labels: Labels tensor, float - [batch_size, 2].

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    with tf.name_scope('Angular_error') as scope:
        numerator = tf.reduce_sum(tf.multiply(output, labels), axis=1)
        denominator = tf.multiply(
            tf.sqrt(tf.reduce_sum(tf.multiply(output, output), axis=1)),
            tf.sqrt(tf.reduce_sum(tf.multiply(labels, labels), axis=1)))
        angular_loss = tf.reduce_mean(tf.acos(tf.div(numerator, denominator)),
                                      name="Angular_loss")
    return angular_loss
