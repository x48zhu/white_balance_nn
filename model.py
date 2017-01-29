import tensorflow as tf
from tensorflow.contrib.layers import flatten

mu = 0
sigma = 0.1

def hyp_net_inference(input):
    # Common Convolutional Layers
    # Layer 1: Input = 47x47x2, Output = 10x10x128
    conv1_W = tf.Variable(tf.truncated_normal(shpae=(8,8,2,128), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(128))
    conv1 = tf.nn.conv2d(input, conv1_W, strides=(1,4,4,1), padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    # conv1 = tf.nn.max_pool(conv1, ksize=[], strides=[], padding='VALID')

    # Layer 2: Input = , Output = 4x4x256
    conv2_W = tf.Variable(tf.truncated_normal(shape=(4,4,128,256), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(256))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=(1,4,4,1), padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    # conv2 = tf.nn.max_pool(conv2, ksize=[], strides=[], padding='VALID')

    # Flatten: Input = 4x4x256, Output = 4096
    fc0 = flatten(conv2)

    # Branch A Full Connected Layer
    fc1A_W = tf.Variable(tf.truncated_normal(shpae=(4096, 256), mean = mu, stddev = sigma))
    fc1A_b = tf.Variable(tf.zeros(256))
    fc1A = tf.matmul(fc0, fc1A_W) + fc1A_b
    fc1A = tf.nn.relu(fc1A)

    fc2A_W = tf.Variable(tf.truncated_normal(shape=(256, 2), mean = mu, stddev = sigma))
    fc2A_b = tf.Variable(tf.zeros(2))
    outputA = tf.matmul(fc1A, fc2A_W) + fc2A_b
    outputA = tf.nn.relu(outputA)

    # Branch B Full Connected Layer
    fc1B_W = tf.Variable(tf.truncated_normal(shpae=(4096, 256), mean = mu, stddev = sigma))
    fc1B_b = tf.Variable(tf.zeros(256))
    fc1B = tf.matmul(fc0, fc1B_W) + fc1B_b
    fc1B = tf.nn.relu(fc1B)

    fc2B_W = tf.Variable(tf.truncated_normal(shape=(256, 2), mean = mu, stddev = sigma))
    fc2B_b = tf.Variable(tf.zeros(2))
    outputB = tf.matmul(fc1A, fc2B_W) + fc2B_b
    outputB = tf.nn.relu(outputB)

    return outputA, outputB

def hyp_net_loss(outputA, outputB, labels):
    # TODO: need regularization???
    # TODO: 1. need debug: maybe try tensor.eval()
    #       2. calculate ground truth scores
    errorA = tf.square(tf.sub(outputA - labels))
    errorB = tf.square(tf.sub(outputB - labels))
    min_error = tf.select(tf.less(errorA, errorB), loss_A, loss_B)
    loss =  tf.reduce_mean(min_error)
    return loss

def hyp_net_training(loss, learning_rate):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(min_loss, global_step=global_step)

def hyp_net_evaluation(outputA, outputB, labels):
    """
    Use L2-distance for error. Consider use angular error later?
    """
    errorA = tf.square(tf.sub(outputA - labels))
    errorB = tf.square(tf.sub(outputB - labels))
    min_error = tf.select(tf.less(errorA, errorB), loss_A, loss_B)
    loss =  tf.reduce_mean(min_error)
    return loss

def calc_ground_truth_score(outputA, outputB, labels):
    errorA = tf.square(tf.sub(outputA - labels))
    errorB = tf.square(tf.sub(outputB - labels))
    zeros = tf.zeros_like(errorA)
    ones = tf.ones_like(errorA)
    chooseA = pack([ones, zeros], axis=1)
    chooseB = pack([zeros, ones], axis=1)
    ground_truth_score = tf.select(tf.less(errorA, errorB), chooseA, chooseB)
    return ground_truth_score



## TODO: look up Tensorflow namespace!!!!
def sel_net_inference(input):
    # Common Convolutional Layers
    # Layer 1: Input = 47x47x2, Output = 10x10x128
    conv1_W = tf.Variable(tf.truncated_normal(shpae=(8,8,2,128), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(128))
    conv1 = tf.nn.conv2d(input, conv1_W, strides=(1,4,4,1), padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    # conv1 = tf.nn.max_pool(conv1, ksize=[], strides=[], padding='VALID')

    # Layer 2: Input = , Output = 4x4x256
    conv2_W = tf.Variable(tf.truncated_normal(shape=(4,4,128,256), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(256))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=(1,4,4,1), padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    # conv2 = tf.nn.max_pool(conv2, ksize=[], strides=[], padding='VALID')

    # Flatten: Input = 4x4x256, Output = 4096
    fc0 = flatten(conv2)

    fc1_W = tf.Variable(tf.truncated_normal(shpae=(4096, 256), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(256))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(256, 2), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(2))
    logits = tf.matmul(fc1A, fc2B_W) + fc2B_b

    return logits

def sel_net_loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    return tf.reduce_mean(cross_entropy)

def sel_net_training(loss, learning_rate):
    tf.summary.scalar('sel_loss', loss)
    min_loss = tf.cond(tf.less(loss_A, loss_B), loss_A, loss_B)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(min_loss, global_step=global_step)

def sel_net_evaluation(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    return tf.reduce_mean(cross_entropy)
