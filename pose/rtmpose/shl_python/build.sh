#!/bin/bash

riscv64-unknown-linux-gnu-gcc shl_wrapper.c ../hhb_out/model.c -fPIC -o model_wrapper.so ../hhb_out/io.c -I ../hhb_out/ -I ../th1520/include/ -L ../th1520/lib/ -lshl -lm -shared
