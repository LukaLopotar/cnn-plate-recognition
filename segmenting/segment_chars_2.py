from skimage.io import imread
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import white_tophat, dilation, disk
from skimage.transform import resize
import numpy as np


def segment_chars(image_path):
    possible_plates = _segment_plate(image_path)

    if len(possible_plates) != 1:
        return 0
    else:
        license_plate = possible_plates[0]

        threshold = threshold_otsu(license_plate)
        license_plate = license_plate > threshold

        license_plate = np.invert(license_plate)

        labelled_plate = label(license_plate)

        characters = []
        for region in regionprops(labelled_plate):
            y0, x0, y1, x1 = region.bbox
            region_height = y1 - y0
            region_width = x1 - x0

            aspect_ratio = region_width / region_height

            if (region.area > 100) and (0.2 < aspect_ratio < 0.9):
                roi = license_plate[y0:y1, x0:x1]

                resized_char = resize(roi, (20, 20))
                characters.append(resized_char)

        return characters


def _segment_plate(image_path):
    # Učitaj sliku:
    car_image = imread(image_path, as_grey=True)

    # Postavi piksele da budu vrijednosti izmedu 0 i 255, umjesto izmedu 0 i 1:
    car_image = car_image * 255

    # White tophat:
    wt_car_image = white_tophat(car_image, disk(10))

    # U binary:
    block_size = 105
    threshold = threshold_local(wt_car_image, block_size, offset=-30)
    binary_image = wt_car_image > threshold

    # Dilation:
    dilated_image = dilation(binary_image, disk(2))

    label_image = label(dilated_image)

    plate_like_objects = []

    for region in regionprops(label_image):
        min_row, min_col, max_row, max_col = region.bbox
        aspect_ratio = (max_col - min_col) / (max_row - min_row)

        if (10000 > region.area > 2000) and (2.2 < aspect_ratio < 4.5):
            plate_like_objects.append(car_image[min_row:max_row, min_col:max_col])

    return plate_like_objects
