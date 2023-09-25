#!/bin/bash

hhb -f mobilenetv2-12.onnx --data-scale 0.017 --data-mean "124 117 104"  --board th1520  --postprocess save_and_top5 --input-name "input" --output-name "output" --input-shape "1 3 224 224" --calibrate-dataset persian_cat.jpg  --quantization-scheme "int8_asym" -D

OPENCV_DIR=../../modules/opencv/
riscv64-unknown-linux-gnu-g++ main.cpp -I${OPENCV_DIR}/include/opencv4 -L${OPENCV_DIR}/lib   -lopencv_imgproc   -lopencv_imgcodecs -L${OPENCV_DIR}/lib/opencv4/3rdparty/ -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2    -lopencv_core -ldl  -lpthread -lrt -lzlib -lcsi_cv -latomic -static -o mobilenetv2_example
