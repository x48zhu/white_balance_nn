import os
import sys
import time

import tensorflow as tf
from model import *
from data import *

######################################################################
# Constants
check_ptr_dir = './save'
if not os.path.exists(check_ptr_dir):
    os.makedirs(check_ptr_dir)
learning_rate = 0.001

EPOCHS = 10
BATCH_SIZE = 128

######################################################################
# Data
train_X, train_y = load_data()
X = tf.placeholder(tf.float32, (None, 47, 47, 2))
y = tf.placeholder(tf.int32, (None))

saver = tf.train.Saver()

######################################################################
# Training pipeline

# HypNet
output_A, output_B = HypNet(X)
# TODO: need regularization???
loss_A = tf.reduce_mean(tf.square(tf.sub(output_A - y)))
loss_B = tf.reduce_mean(tf.square(tf.sub(output_B - y)))
# TODO: 1. need debug: maybe try tensor.eval()
# 		2. calculate ground truth scores
min_loss = tf.cond(tf.less(loss_A, loss_B), loss_A, loss_B)
train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(min_loss)

# SelNet
# TODO: ground_truth_score = ????????????????
logits = SelNet(X)
sel_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, ground_truth_score))
tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sel_loss)

######################################################################

def eval(session):
	pass

def train_hyp_net():
	train_X = hyp_net_pre_process(train_X)
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables()) 

		print ('Training HypNet...')
		for i in range(EPOCHS):
			train_X, train_y = shuffle(train_X, train_y)
			for offset in range(0, num_examples, BATCH_SIZE):
				batch_X, batch_y = X_train[offset:offset+BATCH_SIZE], 
				# For debug use
				# _, loss_value_A, loss_value_B, min_loss_value = sess.run([train_opt, loss_A, loss_B, min_loss], feed_dict={X: batch_X, y: batch_y})
				sess.run(train_opt, feed_dict={X: batch_X, y: batch_y})

		saver.save(sess, './hyp_net')

def train_sel_net():
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		print ('Training SelNet...')
		for i in range(EPOCHS):
			train_X, train_y = shuffle(train_X, train_y)
			for offset in range(0, num_examples, BATCH_SIZE):
				batch_X, batch_y = X_train[offset:offset+BATCH_SIZE], 
				_, loss_value = sess.run([train_opt, loss], feed_dict={X: batch_X, y: batch_y})

		saver.save(sess, './sel_net')
		eval(sess)


if __name__ == "__main__":
	# Sequence of training between the two networks?


