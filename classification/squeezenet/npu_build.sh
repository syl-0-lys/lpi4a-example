#!/bin/bash

hhb -D --model-file squeezenet.caffemodel squeezenet.prototx --data-scale 1 --data-mean "104 117 124"  --board th1520  --postprocess save_and_top5 --input-name "data" --output-name "prob" --input-shape "1 3 227 227" --calibrate-dataset persian_cat.jpg  --quantization-scheme "int8_asym" --pixel-format BGR
OPENCV_DIR=../../modules/opencv/
riscv64-unknown-linux-gnu-g++ main.cpp -I${OPENCV_DIR}/include/opencv4 -L${OPENCV_DIR}/lib   -lopencv_imgproc   -lopencv_imgcodecs -L${OPENCV_DIR}/lib/opencv4/3rdparty/ -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2    -lopencv_core -ldl  -lpthread -lrt -lzlib -lcsi_cv -latomic -static -o squeezenet_example
