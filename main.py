from segmenting.segment_chars import segment_chars
import tensorflow as tf
import numpy as np
from sys import exit

char_to_index_dict = {
    'A': 0, 'B': 1, 'C': 2, 'Č': 3, 'Ć': 4, 'D': 5, 'Đ': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12, 'K': 13,
    'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'R': 19, 'S': 20, 'Š': 21, 'T': 22, 'U': 23, 'V': 24, 'Z': 25, 'Ž': 26,
    'X': 27, 'Y': 28, 'Q': 29, 'W': 30, '0': 31, '1': 32, '2': 33, '3': 34, '4': 35, '5': 36, '6': 37, '7': 38, '8': 39,
    '9': 40
}
img_rows, img_cols = 20, 20

if __name__ == "__main__":
    image_to_recognize_path = "data/vlastite_slike/IMG_20180213_155856.jpg"

    # Segmentiraj znakove:
    characters = segment_chars(image_to_recognize_path)
    if characters == 0:
        print("Nisu nadeni znakovi. Izlazim iz programa.")
        exit(1)

    # Učitaj model konvolucijske mreže:
    with tf.Session() as session:
        # Učitaj izgled grafa:
        saver = tf.train.import_meta_graph("saved_model/cnn_model.meta")

        # Učitaj vrijednosti parametara:
        saver.restore(session, tf.train.latest_checkpoint("saved_model/"))

        # Ucitaj tensore koji su potrebni za ubacivanje slike kroz model i prikaz izlaza modela:
        graph = tf.get_default_graph()
        input_image = graph.get_tensor_by_name("input_image:0")
        keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
        output = graph.get_tensor_by_name("fc2/output:0")

        # Printaj rjesenje:
        for counter, character in enumerate(characters):
            reshaped = np.reshape(character, (1, img_rows * img_cols))

            feed_dict = {input_image: reshaped, keep_prob: 1.0}

            predicted_char = session.run(tf.argmax(output, 1), feed_dict)

            key_predicted = [key for key, value in char_to_index_dict.items() if value == predicted_char][0]

            print("Znak {0}: {1}".format(counter + 1, key_predicted))
