# YOLOv4 for Parkfinder project

To prepare the environment for inference launch, you need to install dependencies from _requirements.txt_ and execute the _install.sh_ script. 

### YOLOv4 code
scripts/inference.py – the simple_inference() function generates an image with bounding boxes and saves it to the _/darknet_ folder. (OUTDATED)

Full inference available via inference() function which allows to process images from the desired folder and cast the output (.csv and/or images with bounding boxes) to the defined directory.

### Occupancy detection 
Detection function get_occupancy() from scripts/utils.py requires as an input .csv produced by the inference() function, .json annotations of parking lots and image metadata from the annotation widget.   

### Annotation widget

A widget for annotating of parking lots images inside the Jupyter Notebook environment. 



Did you miss me?
