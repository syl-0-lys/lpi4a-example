/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/* auto generate by HHB_VERSION "2.3.0" */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <libgen.h>
#include <unistd.h>
#include "io.h"
#include "shl_c920.h"
#include "process.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define FILE_LENGTH 1028
#define SHAPE_LENGHT 128

void *csinn_(char *params);
void csinn_update_input_and_run(struct csinn_tensor **input_tensors, void *sess);
void *csinn_nbg(const char *nbg_file_name);

int input_size[] = {
    1 * 3 * 640 * 640,
};
const char model_name[] = "network";

#define RESIZE_HEIGHT 384
#define RESIZE_WIDTH 640
#define CROP_HEGHT 384
#define CROP_WIDTH 640
#define R_MEAN 0.0
#define G_MEAN 0.0
#define B_MEAN 0.0
#define SCALE 0.003921568627


void *create_graph(char *params_path)
{
    int binary_size;
    char *params = get_binary_from_file(params_path, &binary_size);
    if (params == NULL)
    {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0)
    {
        // create general graph
        return csinn_(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0)
    {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset)
        {
            return csinn_import_binary_model(params);
        }
        else
        {
            return csinn_(params + section->params_offset * 4096);
        }
    }
    else
    {
        return NULL;
    }
}


int main(int argc, char **argv)
{
    char **data_path = NULL;
    int input_num = 1;
    int input_group_num = 1;
    
    int i, j, k;
    uint64_t start_time, end_time;

    if (argc < (2 + input_num))
    {
        printf("Please set valide args: ./model.elf model.params "
               "[tensor1/image1 ...] [tensor2/image2 ...]\n");
        return -1;
    }
    else
    {
        if (argc == 3 && get_file_type(argv[2]) == FILE_TXT)
        {
            data_path = read_string_from_file(argv[2], &input_group_num);
            input_group_num /= input_num;
        }
        else
        {
            data_path = argv + 2;
            input_group_num = (argc - 2) / input_num;
        }
    }

    void *sess = create_graph(argv[1]);

    struct csinn_tensor *input_tensors[input_num];
    input_tensors[0] = csinn_alloc_tensor(NULL);
    input_tensors[0]->dim_count = 4;
    input_tensors[0]->dim[0] = 1;
    input_tensors[0]->dim[1] = 3;
    input_tensors[0]->dim[2] = 640;
    input_tensors[0]->dim[3] = 640;

    float *inputf[input_num];
    int8_t *input[input_num];
    void *input_aligned[input_num];

    for (i = 0; i < input_num; i++)
    {
        input_size[i] = csinn_tensor_byte_size(((struct csinn_session *)sess)->input[i]);
        input_aligned[i] = shl_mem_alloc_aligned(input_size[i], 0);
    }


    for (i = 0; i < input_group_num; i++)
    {
/* set input */
        for (j = 0; j < input_num; j++)
        {
            if (get_file_type(data_path[i * input_num + j]) != FILE_BIN) {
                printf("Please input binary files, since you compiled the model without preprocess.\n");
                return -1;
            }
            
            inputf[j] = (float*)get_binary_from_file(data_path[i * input_num + j], NULL);

            input[j] = shl_ref_f32_to_input_dtype(j, inputf[j], sess);

            memcpy(input_aligned[j], input[j], input_size[j]);
            input_tensors[j]->data = input_aligned[j];
        }

//run
        start_time = shl_get_timespec();
        
        csinn_update_input_and_run(input_tensors, sess);

        end_time = shl_get_timespec();
        printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time - start_time)) / 1000000,
               1000000000.0 / ((float)(end_time - start_time)));


        for (int j = 0; j < input_num; j++)
        {
            shl_mem_free(inputf[j]);
            shl_mem_free(input[j]);
        }
    }

    for (int j = 0; j < input_num; j++)
    {
        csinn_free_tensor(input_tensors[j]);
        shl_mem_free(input_aligned[j]);
    }

    csinn_session_deinit(sess);
    csinn_free_session(sess);

    return 0;
}
