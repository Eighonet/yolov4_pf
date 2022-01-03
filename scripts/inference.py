import os, imghdr, json
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import pandas as pd

from .utils import get_data, draw_boxes, write_images

def simple_inference(threshold = 0.05, input_folder = "yolov4/examples/", image = "parking_test.jpg") -> None:
    inference = "./darknet detector test\
     ../yolov4/data/obj.data ../yolov4/cfg/yolov4-obj.cfg ../yolov4/weights/yolov4-obj_best.weights ../"\
    + path_to_image + image + " -thresh " + str(threshold)
    
    os.system("cd darknet && " + inference)

def inference(threshold = 0.35, input_folder = "yolov4/examples/", output_folder = "yolov4/output/", output_type = "full") -> None:
    images = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and imghdr.what(input_folder + f)]
    print("Detected images in folder: ", images)
    with open('darknet/images_path.txt', 'w') as f:
        for image in images:
            f.write("../" + input_folder + image + "\n")
    inference = "./darknet detector test \
        ../yolov4/data/obj.data ../yolov4/cfg/yolov4-obj.cfg ../yolov4/weights/yolov4-obj_best.weights "\
        + " -thresh " + str(threshold) + " -ext_output -dont_show -out result.json < images_path.txt"
    os.system("cd darknet && " + inference)
    
    with open('darknet/result.json', 'r') as f:
        distros_dict_train = json.load(f)
        result = np.array(get_data(distros_dict_train))
        
    result = np.array(get_data(distros_dict_train))
    cars_coords = result[:, 0] - 1
    cars_coords = np.vstack((cars_coords, result[:, 6]))
    cars_coords = np.vstack((cars_coords, result[:, 2] - result[:, 4]/2))
    cars_coords = np.vstack((cars_coords, result[:, 2] + result[:, 4]/2))
    cars_coords = np.vstack((cars_coords, result[:, 3] - result[:, 5]/2))
    cars_coords = np.vstack((cars_coords, result[:, 3] + result[:, 5]/2))
    cars_coords = cars_coords.T
    
    target_output_size = 600 # scaling height for each output image
    
    if output_type == "full":
        csv_out = pd.DataFrame(cars_coords, columns = ["image_id", "confidence", "x_left", "x_right", "y_bottom", "y_top"])
        csv_out.to_csv(output_folder + "output.csv")
        write_images(input_folder, output_folder, images, target_output_size, cars_coords)
        
    if output_type == "images":
        write_images(input_folder, output_folder, images, target_output_size, cars_coords)
        
    if output_type == "csv":
        csv_out = pd.DataFrame(cars_coords, columns = ["image_id", "confidence", "x_left", "x_right", "y_bottom", "y_top"])
        csv_out.to_csv(output_folder + "output.csv")

            
    
    
