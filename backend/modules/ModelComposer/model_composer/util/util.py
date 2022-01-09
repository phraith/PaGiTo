import numpy as np
import json


def concat_vert_lps(lps, edf_img):
    res = [extract_lp_vert(edf_img, lp) for lp in lps]
    intensities = [elem[0] for elem in res]
    indices = [elem[1] for elem in res]

    concat_intensities = np.concatenate(
        intensities, axis=None)

    concat_indices = np.concatenate(
        indices, axis=None)

    return concat_intensities, concat_indices


def extract_lp_hor(edf_image, y):
    edf_shape = edf_image.shape
    edf_shape_x = edf_shape[1]

    start_idx = edf_shape_x * y
    return edf_image.data[y, :].squeeze(), np.arange(start_idx, start_idx + edf_shape_x)


def extract_lp_vert(edf_image, x):
    edf_shape = edf_image.shape
    edf_shape_y = edf_shape[0]
    edf_shape_x = edf_shape[1]

    end_idx = edf_shape_y * edf_shape_x + x
    return edf_image.data[:, x].squeeze(), np.arange(x, end_idx, edf_shape_x)


def extract_lp_of_size(edf_image, size):
    flattened_image = edf_image.data.flatten()
    return flattened_image[:size], np.arange(0, size)


def convert_image_to_1D(edf_image):
    edf_shape = edf_image.shape
    edf_shape_y = edf_shape[0]
    edf_shape_x = edf_shape[1]

    data = edf_image.data

    return data.flatten(), np.arange(0, edf_shape_x * edf_shape_y)


def get_config(path_to_config):
    with open(path_to_config, 'r') as myfile:
        config_string_data = myfile.read().replace('\n', '')
        try:
            json_file = json.loads(config_string_data)

        except json.JSONDecodeError:
            print("String could not be converted to JSON")

    return json_file, config_string_data


def flip(intensities):
    arr = np.array(intensities)
    arr = np.flip(arr)
    sim_int = arr.tolist()

    return sim_int
