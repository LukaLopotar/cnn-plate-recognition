import cv2
import os
import numpy as np

def fill_with_data_for_recognition(images_path, image_width, image_height):
    images_names = os.listdir(images_path)
    num_images = len(images_names)

    array_images = np.ndarray((num_images, image_height, image_width), np.float32)

    img_index = 0
    for filename in images_names:
        image_full_path = images_path + filename
        image = cv2.imread(image_full_path)

        # Promijeni sliku u sivo:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalziraj piksele da budu između 0 i 1:
        image = image.astype(np.float32) / 255

        array_images[img_index] = image

        img_index += 1

    return array_images


def fill_labels_with_data(file_path, num_images, char_dict, array):
    num_possible_val_for_char = len(char_dict.keys())

    with open(file_path) as f:
        tablica = [line.rstrip('\n') for line in f]

    for x in range(num_images):
        # image = mpimg.imread(full_img_path)

        # plt.imshow(image)

        one_hot_vector = np.zeros(num_possible_val_for_char, np.float32)
        one_hot_vector[char_dict[tablica[x]]] = 1.

        array[x] = one_hot_vector

    return array
