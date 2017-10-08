import tensorflow as tf

WINDOW_SHAPE = (64, 128)
def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=(1, stride[0], stride[1], 1),
                        padding=padding)

def generate_model():
    x = tf.placeholder(tf.float32, shape=(None, None, None),
                       name="input")

    # 1st Convolution layer
    conv_1_w = tf.Variable(tf.truncated_normal((5, 5, 1, 48), stddev=0.1),
                           name="Conv1-Weights")
    conv_1_b = tf.Variable(tf.constant(0.1, shape=(48, )),
                           name="Conv1-bias")
    x_expanded = tf.expand_dims(x, axis=3)
    conv_1_h = tf.nn.relu(conv2d(x_expanded, conv_1_w) + conv_1_b,
                          name="relu_1")
    pool_1_h = tf.nn.max_pool(conv_1_h, ksize=(1, 2, 2, 1),
                              strides=(1, 2, 2, 1), padding="SAME",
                              name="max_pool_1")

    # 2nd Convolution layer
    conv_2_w = tf.Variable(tf.truncated_normal((5, 5, 48, 64), stddev=0.1),
                           name="Conv2-Weights")
    conv_2_b = tf.Variable(tf.constant(0.1, shape=(64, )),
                           name="Conv2-bias")

    conv_2_h = tf.nn.relu(conv2d(pool_1_h, conv_2_w) + conv_2_b,
                          name="relu_2")
    pool_2_h = tf.nn.max_pool(conv_2_h, ksize=(1, 2, 1, 1),
                              strides=(1, 2, 1, 1), padding="SAME",
                              name="max_pool_2")

    # 3rd Convolution layer
    conv_3_w = tf.Variable(tf.truncated_normal((5, 5, 64, 128), stddev=0.1),
                           name="Conv3-Weights")
    conv_3_b = tf.Variable(tf.constant(0.1, shape=(128, )),
                           name="Conv3-bias")

    conv_3_h = tf.nn.relu(conv2d(pool_2_h, conv_3_w) + conv_3_b,
                          name="relu_3")
    pool_3_h = tf.nn.max_pool(conv_3_h, ksize=(1, 2, 2, 1),
                              strides=(1, 2, 2, 1), padding="SAME",
                              name="max_pool_3")

    # flatten Convolution Layer
    flattened = tf.reshape(pool_3_h, [-1, 32 * 8 * 128],
                           name="flatten")

    # dense layer
    fc_1_w = tf.Variable(tf.truncated_normal((32 * 8 * 128, 1024), stddev=0.1),
                           name="fully_connected1-weights")
    fc_1_b = tf.Variable(tf.constant(0.1, shape=(1024, )),
                           name="fully_connected1-bias")

    fc_1_h = tf.nn.relu(tf.matmul(flattened, fc_1_w) + fc_1_b,
                        name="relu_4")

    # Output layer
    fc_2_w = tf.Variable(tf.truncated_normal((1024, 8), stddev=0.1),
                           name="fully_connected2-weights")
    fc_2_b = tf.Variable(tf.constant(0.1, shape=(8, )),
                           name="fully_connected2-bias")

    y = tf.matmul(fc_1_h, fc_2_w) + fc_2_b
    conv_vars = [conv_1_w, conv_1_b, conv_1_h, pool_1_h,
                 conv_2_w, conv_2_b, conv_2_h, pool_2_h,
                 conv_3_w, conv_3_b, conv_3_h, pool_3_h]
    return x, y, conv_vars + [fc_1_w, fc_1_b, fc_2_w, fc_2_b]