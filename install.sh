wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YocbelRjhwUGYcIXxQsWzwVK4UbKCI01' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YocbelRjhwUGYcIXxQsWzwVK4UbKCI01" -O yolov4-obj_best.weights && rm -rf /tmp/cookies.txt
mkdir yolov4/weights
mv yolov4-obj_best.weights yolov4/weights/yolov4-obj_best.weights
git clone https://github.com/AlexeyAB/darknet darknet
cd darknet
make
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
