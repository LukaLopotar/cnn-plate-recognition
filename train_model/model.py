import tensorflow as tf

img_width = 20
img_height = 20
img_depth = 1

conv1_filter_size = 5
conv1_filter_num = 16
conv1_inputs = 1

conv2_filter_size = 5
conv2_filter_num = 32
conv2_inputs = conv1_filter_num

fc1_num_outputs = 256

fc2_num_inputs = fc1_num_outputs
fc2_num_outputs = 41


def model(input_image):
    with tf.name_scope('reshape'):
        input_image = tf.reshape(input_image, [-1, img_width, img_height, img_depth])

    with tf.name_scope('conv1'):
        # 1. konvolucijski sloj:
        W_conv1 = weight_variable([conv1_filter_size, conv1_filter_size, conv1_inputs, conv1_filter_num])
        b_conv1 = bias_variable([conv1_filter_num])
        h_conv1 = tf.nn.relu(conv2d(input_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        # 1. sloj sažimanja:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        # 2. kovolucijski sloj:
        W_conv2 = weight_variable([conv2_filter_size, conv2_filter_size, conv2_inputs, conv2_filter_num])
        b_conv2 = bias_variable([conv2_filter_num])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        # 2. sloj sažimanja:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('dropout'):
        # Dropout:
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_pool2_dropout = tf.nn.dropout(h_pool2, keep_prob)

    # dimenzija slike u ovom trenutku je 5x5xbroj filtera iz prošlog sloja):
    with tf.name_scope('fc1'):
        # 1. potpuno spojeni sloj:
        W_fc1 = weight_variable([5 * 5 * conv2_filter_num, fc1_num_outputs])
        b_fc1 = bias_variable([fc1_num_outputs])

        h_dropout_flat = tf.reshape(h_pool2_dropout, [-1, 5 * 5 * conv2_filter_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_dropout_flat, W_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([fc2_num_inputs, fc2_num_outputs])
        b_fc2 = bias_variable([fc2_num_outputs])

        y_conv = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name="output")

    return y_conv, keep_prob


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


def max_pool_5x5(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding="SAME")
