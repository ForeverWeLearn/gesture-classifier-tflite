from itertools import chain
from string import ascii_letters
from random import choices
from copy import deepcopy
import numpy as np
import cv2


def generate_random_string(n: int):
    return ''.join(choices(ascii_letters, k=n))


def calc_bounding_box(points):
    x, y, w, h = cv2.boundingRect(np.array(points))
    return [x, y, x + w, y + h]


def calc_keypoints(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    keypoints = []

    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)

        keypoints.append([x, y])

    return keypoints


def normalize_keypoints(keypoints):
    temp_landmark_list = deepcopy(keypoints)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for i, landmark_point in enumerate(temp_landmark_list):
        if i == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[i][0] = temp_landmark_list[i][0] - base_x
        temp_landmark_list[i][1] = temp_landmark_list[i][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
