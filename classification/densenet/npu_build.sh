#!/bin/bash

hhb -f densenet.pb --data-scale 0.017 --data-mean "124 117 104"  --board th1520  --postprocess save_and_top5 --input-name "Placeholder" --output-name "densenet121/predictions/Reshape_1" --input-shape "1 224 224 3" --calibrate-dataset persian_cat.jpg  --quantization-scheme "int8_asym" -D

OPENCV_DIR=../../modules/opencv/
riscv64-unknown-linux-gnu-g++ main.cpp -I${OPENCV_DIR}/include/opencv4 -L${OPENCV_DIR}/lib   -lopencv_imgproc   -lopencv_imgcodecs -L${OPENCV_DIR}/lib/opencv4/3rdparty/ -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2    -lopencv_core -ldl  -lpthread -lrt -lzlib -lcsi_cv -latomic -static -o densenet_example
