import sys
import time

import tensorflow as tf
from model import *
from data import *
from constant import *

FLAGS = None

def train_hyp_net():
    train_X, train_y = load_data()

    num_examples = len(train_X)
    num_batches_per_epoch = dataset.num_examples_per_epoch() / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    with tf.Graph().as_default():

        learning_rate = FLAGS.init_learning_rate
        global_step = tf.get_variable('global_step', [], trainable=False,
                                      initializer=tf.constant_initializer(0))
        # learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
        #                                           global_step,
        #                                           decay_steps,
        #                                           FLAGS.learning_rate_decay_factor,
        #                                           staircase=True)

        X = tf.placeholder(tf.float32, (None, 47, 47, 2))
        y = tf.placeholder(tf.int32, (None))
        outputA, outputB = hyp_net_inference(X)
        loss = hyp_net_loss(outputA, outputB, y)
        train_op = hyp_net_training(loss, learning_rate)
        eval_correct = hyp_net_evaluation()

        init = tf.initialize_all_variables()
        sess = tf.Session()
        saver = tf.train.Saver()

        tf.scalar_summary("loss", loss)

        train_X = hyp_net_pre_process(train_X)
        with tf.Session() as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            merged = tf.summary.merge_all()

            print ('Training HypNet...')
            for i in range(FLAGS.epochs):
                train_X, train_y = shuffle(train_X, train_y)
                for j in range(0, num_batches_per_epoch):
                    offset = j * FLAGS.batch_size
                    start_time = time.time()
                    batch_X, batch_y = train_X[offset:offset+FLAGS.batch_size], #####?????
                    _, loss_value, summary = sess.run([train_op, loss, merged], feed_dict={X: batch_X, y: batch_y})
                    duration = time.time() - start_time

                
                train_writer.add_summary(summary, i)
            saver.save(sess, './hyp_net')
            summary_writer.close()

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
    train_hyp_net()

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

