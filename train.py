import argparse
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sys
import time

from data import *
from model import *
from constants import *

FLAGS = None

learning_rate = 0.0001
EPOCHS = 30
BATCH_SIZE = 32
PATCH_SIZE = (47, 47, 3)

# The following operations are executed on patch basis. Ensure input X are in IMG_SIZE x IMG_SIZE patches
X = tf.placeholder(tf.float32, (None, PATCH_SIZE[0], PATCH_SIZE[1], 3), name="Images")
y = tf.placeholder(tf.float32, (None, 2), name="Labels")
ground_truth_score = tf.placeholder(tf.float32, (None, 2), name="Ground_Truth_Score")

outputA, outputB = hyp_net_inference(X)
hyp_loss = hyp_net_loss(outputA, outputB, y)
hyp_train_op = hyp_net_training(hyp_loss, learning_rate)
hyp_eval_correct = hyp_net_evaluation(outputA, outputB, y)
calc_gt_score = calc_ground_truth_score(outputA, outputB, y)

output_sel = sel_net_inference(X)
sel_loss = sel_net_loss(output_sel, ground_truth_score)
sel_train_op = sel_net_training(sel_loss, learning_rate)
sel_eval_correct = sel_net_loss(output_sel, ground_truth_score)

output_patch = inference(outputA, outputB, output_sel)

# The following operation is only used for test data set to evaluate the global estimation on an whole image, in which
# case the input patches X are from the same image.
# As in "Deep Specialized Network for Illuminant Estimation", the Local to Global Estimation simply performing a median
# pooling on all the local illuminant estimates of the image, without resort- ing to additional learning,
# In the future, if necessary, consider using SVR-RBF in "Single and Multiple Illuminant Estimation Using Convolutional
# Neural Networks"
# output_image = np.median(output_patch, axis=1)
# numerator = num(output_image, y)
# denominator = denom(output_image, y)
# angular_error_op = angular_error(numerator, denominator)


def evaluate(X_data, y_data):
    angular_losses = []
    sess = tf.get_default_session()

    # For every evaluate image, split it into patches
    for i in range(len(X_data)):
        patches, labels = np.array(split_to_patches(X_data[i], y_data))
        local_estimation = sess.run(output_patch, feed_dict={X: patches, y: labels})
        angular_loss = angular_error_scalar(local_estimation.eval(), labels)
        angular_losses.append(angular_loss)
    return angular_losses


def training(images, labels):
    assert (len(images) == len(labels))
    run_name = datetime.now().strftime("%I:%M%p on %B %d, %Y")
    train_X, test_X, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=0)

    train_X, train_y = split_to_patches(train_X, train_y)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=0)
    learning_rate = FLAGS.learning_rate

    num_batches_per_epoch = len(train_X) // FLAGS.batch_size
    print("learning rate: {0}".format(learning_rate))

    with tf.Graph().as_default():

        global_step = tf.get_variable('global_step', [], trainable=False, initializer=tf.constant_initializer(0))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Summaries
        tf.summary.scalar("Hyp_Training_Loss", hyp_loss)
        tf.summary.scalar("Sel_Training_Loss", sel_loss)

        with tf.Session() as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter(os.path.join(log_dir, run_name), sess.graph)
            merged = tf.summary.merge_all()

            print('Training HypNet...')
            non_improve_count = 0
            min_error = sys.maxsize
            for i in range(FLAGS.epochs):
                train_X, train_y = shuffle(train_X, train_y)
                for j in range(num_batches_per_epoch):
                    offset = j * FLAGS.batch_size
                    start_time = time.time()
                    batch_X, batch_y = train_X[offset:offset + FLAGS.batch_size], \
                                       train_y[offset:offset + FLAGS.batch_size]
                    _, summary = sess.run([hyp_train_op, merged],
                                          feed_dict={X: batch_X, y: batch_y, ground_truth_score: batch_y})
                    duration = time.time() - start_time
                loss_value = sess.run(hyp_eval_correct, feed_dict={X: val_X, y: val_y})
                loss_value = loss_value.item()  # convert to float
                print("Epoch: %d, Hyp Validation loss: %f" % (i + 1, loss_value))

                if loss_value < min_error:
                    min_error = loss_value
                else:
                    non_improve_count += 1
                if non_improve_count > FLAGS.early_stop_threshold:
                    break

                summary_writer.add_summary(summary, i)
            a, b = sess.run([outputA, outputB], feed_dict={X: val_X, y: val_y})
            print("%s \n%s" % (a, b))

            loss_value = sess.run(hyp_eval_correct, feed_dict={X: test_X, y: test_y})
            # Note the Test set loss is the lease square error based on the best estimation of A and B
            print("Finish training HypNet. Test set loss: %s" % loss_value)

            print('Training SelNet...')
            non_improve_count = 0
            min_error = sys.maxsize
            gt_score_train = sess.run(calc_gt_score, feed_dict={X: train_X, y: train_y})
            gt_score_val = sess.run(calc_gt_score, feed_dict={X: val_X, y: val_y})
            for i in range(FLAGS.epochs):
                train_X, gt_score_train = shuffle(train_X, gt_score_train)
                for j in range(num_batches_per_epoch):
                    offset = j * FLAGS.batch_size
                    start_time = time.time()
                    batch_X, batch_gt_score = train_X[offset:offset + FLAGS.batch_size], \
                                              gt_score_train[offset:offset + FLAGS.batch_size]
                    _, summary = sess.run([sel_train_op, merged],
                                          feed_dict={X: batch_X, y: batch_gt_score, ground_truth_score: batch_gt_score})
                    duration = time.time() - start_time
                loss_value = sess.run(sel_eval_correct,
                                      feed_dict={X: val_X, y: gt_score_val, ground_truth_score: gt_score_val})
                loss_value = loss_value.item()  # convert to float
                print("Epoch: %d, Sel Validation loss: %f" % (i + 1, loss_value))

                if loss_value < min_error:
                    min_error = loss_value
                else:
                    non_improve_count += 1
                if non_improve_count > FLAGS.early_stop_threshold:
                    break

                summary_writer.add_summary(summary, i)
            saver.save(sess, check_ptr_dir)
            # Check on the patch output on one test image
            a, b, s, o = sess.run([outputA, outputB, output_sel, output_patch], feed_dict={X: test_X[0], y: test_y[0]})
            print("%s \n%s \n%s \n%s" % (a, b, s, o))
            angular_errors = evaluate(test_X, test_y)
            mean_error = np.mean(angular_errors)
            median_error = np.median(angular_errors)
            print("########## Comparison by angular error: Mean is %s, Median is %s" % (mean_error, median_error))
            summary_writer.close()


def main(_):
    data_set_name = 'Canon5D'
    images, labels = load_data(data_set_name)
    training(images, labels)


if __name__ == "__main__":
    # Sequence of training between the two networks?
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
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
        default=64,
        help='Number of data item in a batch.'
    )

    parser.add_argument(
        '--early_stop_threshold',
        type=int,
        default=5,
        help='Number of epoch without improvement to early stop'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
