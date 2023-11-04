#!/bin/bash

riscv64-unknown-linux-gnu-gcc shl_wrapper.c ../hhb_out/model.c -fPIC -o model_wrapper.so ../hhb_out/io.c -I ../hhb_out/ -I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/ -I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/shl_public -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ -lshl -lm -shared
