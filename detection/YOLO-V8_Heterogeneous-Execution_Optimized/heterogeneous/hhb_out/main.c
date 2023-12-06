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

/* auto generate by HHB_VERSION "2.4.5" */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <libgen.h>
#include <unistd.h>
#include "io.h"
#include "shl_ref.h"
#include "shl_c920.h"

#define MIN(x, y)           ((x) < (y) ? (x) : (y))
#define FILE_LENGTH         1028
#define SHAPE_LENGHT        128
#define FILE_PREFIX_LENGTH  (1028 - 2 * 128)

void *csinn_(char *params);
void csinn_update_input_and_run(struct csinn_tensor **input_tensors , void *sess);
#define csinn_nbg(...) NULL

int input_size[] = {1 * 66 * 6400, 1 * 66 * 40 * 40, 1 * 384 * 20 * 20, };
const char model_name[] = "network";



static void print_tensor_info(struct csinn_tensor *t) {
    printf("\n=== tensor info ===\n");
    printf("shape: ");
    for (int j = 0; j < t->dim_count; j++) {
        printf("%d ", t->dim[j]);
    }
    printf("\n");
    if (t->dtype == CSINN_DTYPE_UINT8) {
        printf("scale: %f\n", t->qinfo->scale);
        printf("zero point: %d\n", t->qinfo->zero_point);
    }
    printf("data pointer: %p\n", t->data);
}


static void postprocess(void *sess, const char *filename_prefix) {
    int output_num, input_num;
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);

    input_num = csinn_get_input_number(sess);
    for (int i = 0; i < input_num; i++) {
        input->data = NULL;
        csinn_get_input(i, input, sess);
        print_tensor_info(input);
    }
    output_num = csinn_get_output_number(sess);
    for (int i = 0; i < output_num; i++) {
        output->data = NULL;
        csinn_get_output(i, output, sess);
        print_tensor_info(output);

        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
        shl_show_top5(foutput, sess);

        #if 0
        char filename[FILE_PREFIX_LENGTH] = {0};
        char shape[SHAPE_LENGHT] = {0};
        shape2string(output->dim, output->dim_count, shape, SHAPE_LENGHT);
        snprintf(filename, FILE_PREFIX_LENGTH, "%s_output%u_%s.txt", filename_prefix, i, shape);
        int output_size = csinn_tensor_size(foutput);
        float* fdata = shl_c920_output_to_f32_dtype(i, output->data, sess);
        save_data_to_file(filename, fdata, output_size);
        shl_mem_free(fdata);
        #endif

        shl_ref_tensor_transform_free_f32(foutput);
        if (!output->is_const) {
            shl_mem_free(output->data);
        }
    }
    csinn_free_tensor(input);
    csinn_free_tensor(output);
}

void *create_graph(char *params_path) {
    char *params = get_binary_from_file(params_path, NULL);
    if (params == NULL) {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0) {
        // create general graph
        return csinn_(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0) {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset) {
            return csinn_import_binary_model(params);
        } else {
            return csinn_(params + section->params_offset * 4096);
        }
    } else {
        return NULL;
    }
}

int main(int argc, char **argv) {
    char **data_path = NULL;
    int input_num = 3;
    int output_num = 1;
    int input_group_num = 1;
    int i;

    if (argc == 3 && get_file_type(argv[2]) == FILE_TXT) {
        data_path = read_string_from_file(argv[2], &input_group_num);
        input_group_num /= input_num;
    } else if (argc >= (2 + input_num)) {
        data_path = argv + 2;
        input_group_num = (argc - 2) / input_num;
    } else {
        printf("Please set valide args: ./model.elf hhb.bm "
                "[data1 data2 ...]|[.txt]\n");
        return -1;
    }

    

    void *sess = create_graph(argv[1]);

    struct csinn_tensor* input_tensors[input_num];
    input_tensors[0] = csinn_alloc_tensor(NULL);
    input_tensors[0]->dim_count = 3;
    input_tensors[0]->dim[0] = 1;
    input_tensors[0]->dim[1] = 66;
    input_tensors[0]->dim[2] = 6400;
    input_tensors[1] = csinn_alloc_tensor(NULL);
    input_tensors[1]->dim_count = 4;
    input_tensors[1]->dim[0] = 1;
    input_tensors[1]->dim[1] = 66;
    input_tensors[1]->dim[2] = 40;
    input_tensors[1]->dim[3] = 40;
    input_tensors[2] = csinn_alloc_tensor(NULL);
    input_tensors[2]->dim_count = 4;
    input_tensors[2]->dim[0] = 1;
    input_tensors[2]->dim[1] = 384;
    input_tensors[2]->dim[2] = 20;
    input_tensors[2]->dim[3] = 20;

    float *inputf[input_num];
    char filename_prefix[FILE_PREFIX_LENGTH] = {0};
    uint64_t start_time, end_time;
    for (i = 0; i < input_group_num; i++) {
        /* set input */
        for (int j = 0; j < input_num; j++) {
            if (get_file_type(data_path[i * input_num + j]) != FILE_BIN) {
                printf("Please input binary files, since you compiled the model without preprocess.\n");
                return -1;
            }
            inputf[j] = (float*)get_binary_from_file(data_path[i * input_num + j], NULL);

            input_tensors[j]->data = shl_c920_f32_to_input_dtype(j, inputf[j], sess);
        }

        start_time = shl_get_timespec();
        csinn_update_input_and_run(input_tensors, sess);
        end_time = shl_get_timespec();
        printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time-start_time))/1000000,
                    1000000000.0/((float)(end_time-start_time)));

        snprintf(filename_prefix, FILE_PREFIX_LENGTH, "%s", basename(data_path[i * input_num]));
        postprocess(sess, filename_prefix);

        for (int j = 0; j < input_num; j++) {
            shl_mem_free(inputf[j]);
            shl_mem_free(input_tensors[j]->data);
        }
    }

    for (int j = 0; j < input_num; j++) {
        csinn_free_tensor(input_tensors[j]);
    }

    csinn_session_deinit(sess);
    csinn_free_session(sess);

    return 0;
}

