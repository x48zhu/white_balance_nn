import argparse
import sys
import time
from datetime import datetime

import tensorflow as tf
from model import *
from data import *
from constants import *

FLAGS = None

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

learning_rate = 0.001
epochs = 30
batch_size = 32

def train_hyp_net(images, labels):
    assert(len(images) == len(labels))
    run_name = datetime.now().strftime("%I:%M%p on %B %d, %Y")
    train_X, test_X, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=0)
    # train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=0)

    num_examples = len(train_X)
    num_batches_per_epoch = len(train_X) // FLAGS.batch_size
    # decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
    # print ("{0} {1} {2}".format(num_examples, num_batches_per_epoch, num_batches_per_epoch))
    
    with tf.Graph().as_default():

        learning_rate = FLAGS.init_learning_rate
        global_step = tf.get_variable('global_step', [], trainable=False,
                                      initializer=tf.constant_initializer(0))
        # learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
        #                                           global_step,
        #                                           decay_steps,
        #                                           FLAGS.learning_rate_decay_factor,
        #                                           staircase=True)

        X = tf.placeholder(tf.float32, (None, IMG_SIZE, IMG_SIZE, 3), name="Images")
        y = tf.placeholder(tf.float32, (None, 2), name="Labels")
        
        outputA, outputB = hyp_net_inference(X)
        hyp_loss = hyp_net_loss(outputA, outputB, y)
        hyp_train_op = hyp_net_training(hyp_loss, learning_rate)
        eval_correct = hyp_net_evaluation(outputA, outputB, y)
        ground_truth_score = calc_ground_truth_score(outputA, outputB, y)
        
        output = sel_net_inference(X)
        sel_loss = sel_net_loss(output, ground_truth_score)
        sel_train_op = sel_net_training(sel_loss, learning_rate)
        eval_correct = sel_net_loss(output, y)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        saver = tf.train.Saver()
        
        # Summaries
        tf.summary.scalar("Training_Loss", hyp_loss)
        # tf.summary.scalar("Validation_Loss", eval_correct)

        # train_X = hyp_net_pre_process(train_X)
        with tf.Session() as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter(os.path.join(log_dir, run_name), sess.graph)
            merged = tf.summary.merge_all()
            print ('Training HypNet...')
            for i in range(FLAGS.epochs):
                train_X, train_y = shuffle(train_X, train_y)
                print("Epoch: %d" % i)
                for j in range(0, num_batches_per_epoch):
                    offset = j * FLAGS.batch_size
                    start_time = time.time()
                    batch_X, batch_y = train_X[offset:offset + FLAGS.batch_size], train_y[offset:offset + FLAGS.batch_size]
                    _, loss_value, summary = sess.run([hyp_train_op, hyp_loss, merged], feed_dict={X: batch_X, y: batch_y})
                    duration = time.time() - start_time
                    print("Training loss: %s; Takes %d" % (loss_value, duration))
                # loss_value = sess.run(eval_correct, feed_dict={X:val_X, y:val_y})
                # print("Validation loss: %s" % (loss_value))
                summary_writer.add_summary(summary, i)

            saver.save(sess, './hyp_net')
            summary_writer.close()
            
            loss_value = sess.run(eval_correct, feed_dict={X:test_X, y:test_y})
            print("Finish training HypNet. Evaluation loss: %s" % loss_value)
            
            # gt_score = sess.run(ground_truth_score, feed_dict={X:train_X, y:train_y})
            # print ('Training SelNet...')
            # for i in range(FLAGS.epochs):
            #     train_X, train_y = shuffle(train_X, train_y)
            #     print("Epoch: %d" % i)
            #     for j in range(0, num_batches_per_epoch):
            #         offset = j * FLAGS.batch_size
            #         start_time = time.time()
            #         batch_X, batch_y = train_X[offset:offset + FLAGS.batch_size], train_y[offset:offset + FLAGS.batch_size]
            #         _, loss_value, summary = sess.run([hyp_train_op, hyp_loss, merged], feed_dict={X: batch_X, y: batch_y})
            #         duration = time.time() - start_time
            #         print("Loss: %s; Takes %d" % (loss_value, duration))
            #     summary_writer.add_summary(summary, i)



def main(_):
    data_set_name = 'Canon5D'
    images, labels = load_data(data_set_name)
    train_hyp_net(images, labels)

if __name__ == "__main__":
    # Sequence of training between the two networks?
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--init_learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
    )

    parser.add_argument(
      '--epochs',
      type=int,
      default=10,
      help='Number of steps to run trainer.'
    )

    parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Number of steps to run trainer.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

