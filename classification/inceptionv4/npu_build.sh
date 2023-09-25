#!/bin/bash

hhb -D -f inceptionv4.pb --data-scale 0.0039 --data-mean "127.5 127.5 127.5"  --board th1520  --postprocess save_and_top5 --input-name "input" --output-name "InceptionV4/Logits/Predictions" --input-shape "1 299 299 3" --calibrate-dataset persian_cat.jpg  --quantization-scheme "int8_asym"

OPENCV_DIR=../../modules/opencv/
riscv64-unknown-linux-gnu-g++ main.cpp -I${OPENCV_DIR}/include/opencv4 -L${OPENCV_DIR}/lib   -lopencv_imgproc   -lopencv_imgcodecs -L${OPENCV_DIR}/lib/opencv4/3rdparty/ -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2    -lopencv_core -ldl  -lpthread -lrt -lzlib -lcsi_cv -latomic -static -o inceptionv4_example
