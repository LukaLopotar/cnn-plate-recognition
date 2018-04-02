from skimage.io import imread
from skimage.filters import threshold_otsu
import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops


def segment_chars(image_path):
    possible_plates = _segment_plate(image_path)

    if len(possible_plates) != 1:
        return 0

    else:
        license_plate = possible_plates[0]

        threshold = threshold_otsu(license_plate)
        license_plate = license_plate > threshold

        license_plate = np.invert(license_plate)

        labelled_plate = measure.label(license_plate)

        character_dimensions = (
        0.35 * license_plate.shape[0], 1.0 * license_plate.shape[0], 0.05 * license_plate.shape[1],
        0.15 * license_plate.shape[1])

        min_height, max_height, min_width, max_width = character_dimensions

        characters = []
        for region in regionprops(labelled_plate):
            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col
            aspect_ratio = region_width / region_height

            if (region.area > 100) and (0.2 < aspect_ratio < 0.9) and (min_height < region_height < max_height) \
                    and (min_width < region_width < max_width):
                roi = license_plate[min_row:max_row, min_col:max_col]

                resized_char = resize(roi, (20, 20))
                characters.append(resized_char)

        return characters


def _segment_plate(image_path):
    # Učitaj sliku:
    car_image = imread(image_path, as_grey=True)

    # Postavi piksele da budu vrijednosti izmedu 0 i 255, umjesto izmedu 0 i 1:
    car_image = car_image * 255

    threshold_value = threshold_otsu(car_image)
    binary_car_image = car_image > threshold_value

    # CCA:
    label_image = measure.label(binary_car_image)

    plate_dimensions = (0.08 * label_image.shape[0], 0.2 * label_image.shape[0], 0.15 * label_image.shape[1], 0.4 * label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    plate_like_objects = []

    for region in regionprops(label_image):
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        aspect_ratio = region_width / region_height

        if (2000 < region.area < 10000) and (2.2 < aspect_ratio < 4.5) and (min_height <= region_height <= max_height)\
                and (min_width <= region_width <= max_width):
            plate_like_objects.append(car_image[min_row:max_row, min_col:max_col])

    return plate_like_objects
