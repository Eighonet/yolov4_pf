import json
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


def get_data(distros_dict) -> list:
    json_to_variable = []
    for distro in distros_dict:
        if len(distro["objects"]) != 0:
            for obj in range(len(distro["objects"])):
                frame_id = distro["frame_id"]
                class_id = distro["objects"][obj]["class_id"]
                x = distro["objects"][obj]["relative_coordinates"]["center_x"]
                y = distro["objects"][obj]["relative_coordinates"]["center_y"]
                width = distro["objects"][obj]["relative_coordinates"]["width"]
                height = distro["objects"][obj]["relative_coordinates"]["height"]
                confidence = distro["objects"][obj]["confidence"]
                json_to_variable.append(
                    [frame_id, class_id, x, y, width, height, confidence]
                )
    return json_to_variable


def draw_boxes(image: np.array, b_box: np.array, scale_factors: list) -> np.array:
    cv2.rectangle(
        image,
        [int(b_box[2] * scale_factors[1]), int(b_box[4] * scale_factors[0])],
        [int(b_box[3] * scale_factors[1]), int(b_box[5] * scale_factors[0])],
        (36, 255, 12),
        thickness=2,
    )
    cv2.putText(
        image,
        str(round(b_box[1], 2)),
        (int(b_box[2] * scale_factors[1]), int(b_box[4] * scale_factors[0]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (36, 255, 12),
        2,
    )
    return image


def write_images(
    input_folder: str,
    output_folder: str,
    images: list,
    target_output_size: int,
    cars_coords: np.array,
) -> None:
    for i in range(len(images)):
        image = cv2.imread(input_folder + images[i])
        scale_general = target_output_size / image.shape[0]
        width, height = int(image.shape[1] * scale_general), int(
            image.shape[0] * scale_general
        )
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        scale_factors = image.shape[0:2]
        b_boxes = cars_coords[cars_coords[:, 0] == i]
        for b_box in b_boxes:
            image = draw_boxes(image, b_box, scale_factors)
        cv2.imwrite(output_folder + "output_" + images[i], image)


def get_occupancy(
    detections_path: str = "yolov4/output/output.csv",
    parking_annotations: str = "annotation_widget/annotations.json",
    image_metadata: str = "annotation_widget/metadata.json",
    occupation_ratio: float = 0.3,
) -> dict:
    detections = pd.read_csv(detections_path, index_col=0)

    with open(parking_annotations, "r") as f:
        parking_lots = json.load(f)

    with open(image_metadata, "r") as f:
        image_shapes = json.load(f)

    detections_dict = (
        detections.groupby("image_id")[list(detections.columns)[1:]]
        .apply(lambda g: g.values.tolist())
        .to_dict()
    )

    for image in image_shapes:
        image_detections = detections_dict[image]
        image_shape = image_shapes[image]
        for detection in image_detections:
            detection[1] = int(detection[1] * image_shape[1])
            detection[2] = int(detection[2] * image_shape[1])
            detection[3] = int(detection[3] * image_shape[0])
            detection[4] = int(detection[4] * image_shape[0])

    result_dict = {image: [] for image in image_shapes}
    for image_name in image_shapes:
        im_detections = np.array(detections_dict[image_name])[:, 1:]
        im_lots = parking_lots[image_name]

        car_boxes = []

        for i in range(len(im_detections)):
            car_boxes.append(
                [
                    [int(im_detections[i][0]), int(im_detections[i][2])],
                    [int(im_detections[i][1]), int(im_detections[i][2])],
                    [int(im_detections[i][1]), int(im_detections[i][3])],
                    [int(im_detections[i][0]), int(im_detections[i][3])],
                ]
            )

        for i in range(len(im_lots)):
            pts = np.array(
                [
                    [im_lots[i][0][0], im_lots[i][1][0]],
                    [im_lots[i][0][1], im_lots[i][1][1]],
                    [im_lots[i][0][2], im_lots[i][1][2]],
                    [im_lots[i][0][3], im_lots[i][1][3]],
                ]
            )
            occupation_flag = 0
            for j in range(len(car_boxes)):
                cat_poly = Polygon(car_boxes[j])
                lot_poly = Polygon(pts)
                if (
                    cat_poly.intersection(lot_poly).area
                    > occupation_ratio * lot_poly.area
                ):
                    occupation_flag += 1
                    break
            result_dict[image_name].append(occupation_flag)
    return result_dict
