import os
import time

import numpy as np
import tensorflow as tf

from utils import helper_functions as hf
from train_model.model import model

# Znakovi = 30 hrv. znakova + 4 znaka (x, y, w, q) + 10 brojeva:
char_to_index_dict = {
    'A': 0, 'B': 1, 'C': 2, 'Č': 3, 'Ć': 4, 'D': 5, 'Đ': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12, 'K': 13,
    'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'R': 19, 'S': 20, 'Š': 21, 'T': 22, 'U': 23, 'V': 24, 'Z': 25, 'Ž': 26,
    'X': 27, 'Y': 28, 'Q': 29, 'W': 30, '0': 31, '1': 32, '2': 33, '3': 34, '4': 35, '5': 36, '6': 37, '7': 38, '8': 39,
    '9': 40
}

# Hiperparametri mreže:
batch_size = 128
num_epochs = 3001
learn_rate = 0.001
display_after_epoch = 100

# Ostali parametri:
img_cols = 20
img_rows = 20
img_depth = 1  # Slike ce biti sive, pa ne treba img_depth
# Koliko klasa postoji za svaki znak:
num_possible_classes = len(char_to_index_dict.keys())

train_images_path = "../data/train_data/images/"
train_labels_path = "../data/train_data/train_labels.txt"

train_images = os.listdir(train_images_path)
train_data = hf.fill_with_data_for_recognition(train_images_path, img_cols, img_rows)

test_images_path = "../data/test_data/images/"
test_labels_path = "../data/test_data/test_labels.txt"

test_images = os.listdir(test_images_path)
test_data = hf.fill_with_data_for_recognition(test_images_path, img_cols, img_rows)

num_train_images = len(train_images)
# One hot vector za svaki od charactera u tablici:
train_labels = np.zeros((num_train_images, num_possible_classes), np.float32)

num_test_images = len(test_images)
# One hot vector za svaki od charactera u tablici:
test_labels = np.zeros((num_test_images, num_possible_classes), np.float32)


train_labels = hf.fill_labels_with_data(train_labels_path, num_train_images, char_to_index_dict, train_labels)
test_labels = hf.fill_labels_with_data(test_labels_path, num_test_images, char_to_index_dict, test_labels)

input_image = tf.placeholder(tf.float32, shape=[None, img_rows * img_cols * img_depth], name="input_image")
target_label = tf.placeholder(tf.float32, shape=[None, num_possible_classes])

y_conv, keep_prob = model(input_image=input_image)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target_label, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(target_label, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

prosjek_test = []

# Saver za spremanje modela:
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print("Initialized.")

    start_time = time.time()
    print("Vrijeme krenulo.")

    for epoch in range(num_epochs):
        offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_data[offset:(offset + batch_size), :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        batch_data = np.reshape(batch_data, (batch_data.shape[0], img_rows * img_cols))

        if epoch % display_after_epoch == 0:
            train_accuracy = accuracy.eval(feed_dict={
                input_image: batch_data, target_label: batch_labels, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (epoch, train_accuracy))
        train_step.run(feed_dict={input_image: batch_data, target_label: batch_labels, keep_prob: 0.5})

    stop_time = time.time()
    print("Vrijeme stalo.")

    for epoch in range(num_epochs):
        test_offset = (epoch * batch_size) % (test_labels.shape[0] - batch_size)
        test_batch_data = test_data[test_offset:(test_offset + batch_size), :, :]
        test_batch_labels = test_labels[test_offset:(test_offset + batch_size), :]
        test_batch_data = np.reshape(test_batch_data, (test_batch_data.shape[0], img_rows * img_cols))

        if epoch % display_after_epoch == 0:
            test_accuracy = accuracy.eval(feed_dict={input_image: test_batch_data,
                                                     target_label: test_batch_labels, keep_prob: 1.0})
            print('step %d, test accuracy %g' % (epoch, test_accuracy))

            prosjek_test.append(test_accuracy)

    print("PROSJEK USPJESNOSTI: {0:.2f} %".format((sum(prosjek_test)/len(prosjek_test)) * 100))
    print("VRIJEME POTREBNO ZA TRENIRANJE: {0:.2f} sekundi.".format(stop_time - start_time))

    print("Spremam model...")
    saver.save(session, "../saved_model/cnn_model")
