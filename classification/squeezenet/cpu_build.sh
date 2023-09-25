#!/bin/bash

hhb -D --model-file squeezenet.caffemodel squeezenet.prototxt --data-scale 1 --data-mean "104 117 124"  --board c920  --postprocess save_and_top5 --input-name "data" --output-name "prob" --input-shape "1 3 227 227" --quantization-scheme float16 --pixel-format BGR

OPENCV_DIR=../../modules/opencv/
riscv64-unknown-linux-gnu-g++ main.cpp -I${OPENCV_DIR}/include/opencv4 -L${OPENCV_DIR}/lib -L${OPENCV_DIR}/lib/opencv4/3rdparty/ -Wl,--start-group -lopencv_imgproc -lopencv_imgcodecs -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2 -lopencv_core -Wl,--end-group -ldl  -lpthread -lrt -lzlib -lcsi_cv -latomic -static -o squeezenet_example
