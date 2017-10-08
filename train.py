import tensorflow as tf
import model
from time import time
from data_extractor import extract_data, batches_


def __losses__(y, y_):
    return tf.losses.mean_squared_error(y_, y)


def train(learning_rate, n_epochs,
          restore=False, path=None):
    """
    Train the models
    @params:
        learning_rate: Learning Rate (alpha) for the model
        n_epochs: Number of times to go through data
        restore: To restore a previously trained model
        path: Path to the place where the model is either to 
              be saved or to be loaded from
    """
    x, y, params = model.generate_model()
    y_ = tf.placeholder(tf.float32, shape=(None, 8),
                        name="ground_truth")

    loss = __losses__(y, y_)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    x_train, y_train = extract_data('data/train.csv')
    x_test, y_test = extract_data('data/test.csv')
    x_train = x_train.reshape(-1, 64, 128)
    x_test = x_test.reshape(-1, 64, 128)

    saver = tf.train.Saver()

    if restore:
        with tf.Session() as sess:
            saver.restore(sess, path)
            print("\n Model Restored\n\n")
            try:
                for i in range(n_epochs):
                    t = time()
                    for batch_xs, batch_ys in batches_(x_train, y_train):
                        sess.run(train_step,
                                 feed_dict={x: batch_xs, y_: batch_ys})
                    l= sess.run(loss, feed_dict={x: x_test, y_: y_test})
                    print("\nRun {}-------------------Time: {}".format(i, time() - t))
                    print("Loss : {}".format(l))

                save_path = saver.save(sess, path)
                print("Model saved in file: %s" % save_path)
            except:
                sess.close()
            sess.close()
    else:
        with tf.Session() as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            try:
                for i in range(n_epochs):
                    t = time()
                    for batch_xs, batch_ys in batches_(x_train, y_train):
                        sess.run(train_step,
                                 feed_dict={x: batch_xs, y_: batch_ys})
                    l= sess.run(loss, feed_dict={x: x_test, y_: y_test})
                    print("\nRun {}-------------------Time: {}".format(i, time() - t))
                    print("Loss : {}".format(l))

                save_path = saver.save(sess, path)
                print("Model saved in file: %s" % save_path)
            except:
                sess.close()
            sess.close()
        


if __name__ == '__main__':
    train(0.003, 40, path="./model/model.ckpt")