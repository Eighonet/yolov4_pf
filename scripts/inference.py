import cv2, os

def simple_inference(threshold = 0.05, path_to_image = "yolov4/examples/", image = "parking_test.jpg") -> None:
    inference = "./darknet detector test\
     ../yolov4/data/obj.data ../yolov4/cfg/yolov4-obj.cfg ../yolov4/weights/yolov4-obj_best.weights ../"\
    + path_to_image + image + " -thresh" + str(0.05)
    
    os.system("cd darknet && " + inference)