import tensorflow as tf
from tensorflow.contrib.layers import flatten

my = 0
sigma = 0.1

def HypNet(input):
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
	fc2A = tf.matmul(fc1A, fc2A_W) + fc2A_b
	fc2A = tf.nn.relu(fc2A)

	# Branch B Full Connected Layer
	fc1B_W = tf.Variable(tf.truncated_normal(shpae=(4096, 256), mean = mu, stddev = sigma))
	fc1B_b = tf.Variable(tf.zeros(256))
	fc1B = tf.matmul(fc0, fc1B_W) + fc1B_b
	fc1B = tf.nn.relu(fc1B)

	fc2B_W = tf.Variable(tf.truncated_normal(shape=(256, 2), mean = mu, stddev = sigma))
	fc2B_b = tf.Variable(tf.zeros(2))
	fc2B = tf.matmul(fc1A, fc2B_W) + fc2B_b
	fc2B = tf.nn.relu(fc2B)

	return fc2A, fc2B


## TODO: look up Tensorflow namespace!!!!
def SelNet(input):
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
