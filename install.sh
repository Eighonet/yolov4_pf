#!/bin/sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YocbelRjhwUGYcIXxQsWzwVK4UbKCI01' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YocbelRjhwUGYcIXxQsWzwVK4UbKCI01" -O yolov4-obj_best.weights && rm -rf /tmp/cookies.txt
mkdir yolov4/weights
mv yolov4-obj_best.weights yolov4/weights/yolov4-obj_best.weights
