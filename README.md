# YOLOv4 for Parkfinder project

To prepare the environment for inference launch, you need to install dependencies from _requirements.txt_ and execute the _install.sh_ script. 

### Code
scripts/inference.py – the currently implemented simple_inference() function generates an image with bounding boxes and saves it to the _/darknet_ folder.

Full inference available via inference() function which allows to process images from the desired folder and cast the output (.csv and/or images with bounding boxes) to the defined directory.

Did you miss me?
