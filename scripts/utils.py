import os, imghdr, json
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import pandas as pd

def get_data(distros_dict) -> list:
    json_to_variable = []
    for distro in distros_dict:
        filename = distro['filename']
        if len(distro['objects']) != 0:
            for obj in range(len(distro['objects'])):
                frame_id = distro['frame_id']
                class_id = distro['objects'][obj]["class_id"]
                x = distro['objects'][obj]["relative_coordinates"]["center_x"]
                y = distro['objects'][obj]["relative_coordinates"]["center_y"]
                width = distro['objects'][obj]["relative_coordinates"]["width"]
                height = distro['objects'][obj]["relative_coordinates"]["height"]
                confidence = distro['objects'][obj]["confidence"]
                json_to_variable.append([frame_id, class_id, x, y, width, height, confidence])
    return json_to_variable

def draw_boxes(image: np.array, b_box: np.array, scale_factors: list) -> np.array:
    cv2.rectangle(image, [int(b_box[2]*scale_factors[1]), int(b_box[4]*scale_factors[0])],\
                         [int(b_box[3]*scale_factors[1]), int(b_box[5]*scale_factors[0])],\
                         (36,255,12), thickness=2)
    cv2.putText(image, str(round(b_box[1], 2)),\
               (int(b_box[2]*scale_factors[1]), int(b_box[4]*scale_factors[0]) - 10),\
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
    return image

def write_images(input_folder: str, output_folder: str, images: list,\
                target_output_size: int, cars_coords: np.array) -> None:
    for i in range(len(images)):
        image = cv2.imread(input_folder + images[i])
        scale_general = target_output_size/image.shape[0]
        width, height = int(image.shape[1] * scale_general), int(image.shape[0] * scale_general)
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

        scale_factors = image.shape[0:2]
        b_boxes = cars_coords[cars_coords[:, 0] == i]
        for b_box in  b_boxes: 
            image = draw_boxes(image, b_box, scale_factors)
        cv2.imwrite(output_folder + "output_" + images[i], image)
