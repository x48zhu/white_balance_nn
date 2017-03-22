import argparse
from datetime import datetime
import logging
import numpy as np
import os.path
import sys
import tensorflow as tf
import time

import simple.model as model
from simple.data import load_data
from utils import logger, angular_error_scalar
from simple.constants import log_dir

FLAGS = None


def fill_feed_dict(data_set, features_pl, labels_pl, batch_size=None):
    if not batch_size:
        batch_size = FLAGS.batch_size
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {features_pl: images_feed, labels_pl: labels_feed}
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            data_set_name):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: Data set to run the evaluation on.
      data_set_name: The name of the data set, for print purpose
    """
    # And run one epoch of eval.
    angular_error_value = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        angular_error_value += sess.run(eval_correct, feed_dict=feed_dict)
    precision = angular_error_value / num_examples
    logger.info('%s Error: %0.04f' % (data_set_name, precision))


def run_training(debug=False):
    """Train MNIST for a number of steps.

    Args:
        debug: debug mode
    """
    run_name = datetime.now().strftime("%I:%M%p on %B %d, %Y")
    data_sets = load_data(FLAGS.data_set_name, FLAGS.valid_percent,
                          FLAGS.test_percent, debug)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        features_placeholder = tf.placeholder(tf.float32, shape=(None, 8))
        labels_placeholder = tf.placeholder(tf.float32, shape=(None, 2))

        # Build a Graph that computes predictions from the inference model.
        output = model.inference(features_placeholder,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2)

        loss = model.loss(output, labels_placeholder)
        train_op = model.training(loss, FLAGS.learning_rate)
        eval_correct = model.evaluation(output, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        logger.info("Training...")
        with tf.Session() as sess:
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.log_dir, run_name), sess.graph)
            sess.run(init)

            non_improve_count = 0
            min_error = 1e5
            # Start the training loop.
            for step in range(FLAGS.max_steps):
                start_time = time.time()
                feed_dict = fill_feed_dict(data_sets.train,
                                           features_placeholder,
                                           labels_placeholder)

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                duration = time.time() - start_time

                logger.info('Step %d: square mean loss = %.2f (%.3f sec)' % (
                    step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                # if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                do_eval(sess,
                        eval_correct,
                        features_placeholder,
                        labels_placeholder,
                        data_sets.train,
                        "Training")
                # Evaluate against the validation set.
                do_eval(sess,
                        eval_correct,
                        features_placeholder,
                        labels_placeholder,
                        data_sets.valid,
                        "Validation")
                # Evaluate against the test set.
                do_eval(sess,
                        eval_correct,
                        features_placeholder,
                        labels_placeholder,
                        data_sets.test,
                        "Test")

                # if loss_value < min_error:
                #     min_error = loss_value
                # else:
                #     non_improve_count += 1
                # if non_improve_count > FLAGS.early_stop_threshold:
                #     break

            if debug:
                feed_dict = fill_feed_dict(data_sets.test,
                                           features_placeholder,
                                           labels_placeholder,
                                           batch_size=data_sets.test.num_examples)
                predict = sess.run(output, feed_dict=feed_dict)
                ground_truth = data_sets.test.labels
                avg_angular_error = angular_error_scalar(predict, ground_truth)
                logger.info("%f" % avg_angular_error)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if FLAGS.debug:
        logger.setLevel(logging.DEBUG)
    run_training(FLAGS.debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Debug mode.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=16,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--data_set_name',
        type=str,
        default='Canon5D',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=log_dir,
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--test_percent',
        type=int,
        default=0.2,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--valid_percent',
        type=int,
        default=0.2,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--early_stop_threshold',
        type=int,
        default=5,
        help='Number of epoch without improvement to early stop'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
