import sys
import os

sys.path.insert(0, "bert")

import numpy as np

from run_squad import read_squad_examples, convert_examples_to_features
from run_squad import RawResult, write_predictions
import tokenization


max_seq_length = 384
max_query_length = 64
doc_stride = 128


def get_data(vocab_file, predict_file):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    eval_examples = read_squad_examples(input_file=predict_file, is_training=False)

    eval_features = []

    def append_feature(feature):
        eval_features.append(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=append_feature,
    )

    return eval_examples, eval_features


def load_result_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data_list = []
        for line in lines:
            data_list.append(float(line.strip()))
        arr = np.array(data_list)
        return arr

def inference_model(features, model_input_type=np.int64, output_dir=".", save_inter_results=False):
    res = []

    input_list = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, eval_features in enumerate(features):
        fd = {
            "input_ids": np.array(eval_features.input_ids).astype(model_input_type)[
                np.newaxis, :
            ],
            "input_mask": np.array(eval_features.input_mask).astype(model_input_type)[
                np.newaxis, :
            ],
            "segment_ids": np.array(eval_features.segment_ids).astype(model_input_type)[
                np.newaxis, :
            ],
        }

        sample_prefix = f"sample_{idx}_"
        if save_inter_results:
            tmp_list = []
            for k, v in fd.items():
                curr_input = os.path.join(output_dir, sample_prefix + f"{k}.bin")
                v.astype(np.float32).tofile(curr_input)
                tmp_list.append(curr_input)
            input_list.append(" ".join(tmp_list) + "\n")


        model_inference_command = "./hhb_out/hhb_runtime ./hhb_out/hhb.bm sample_0_input_ids.bin  sample_0_segment_ids.bin sample_0_input_mask.bin"
        os.system(model_inference_command)

        unique_id = eval_features.unique_id

        start_logits = load_result_file("sample_0_input_ids.bin_output0_1_384.txt")
        end_logits = load_result_file("sample_0_input_ids.bin_output1_1_384.txt")

        res.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
            )
        )

    return res


def postprocess(eval_examples, eval_features, all_results, output_dir="."):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")

    write_predictions(eval_examples, eval_features, all_results,
        20, 30,
        True, output_prediction_file,
        output_nbest_file, output_null_log_odds_file)


if __name__ == "__main__":
    vocab_file = "vocab.txt"
    predict_file = "test1.json"

    output_path = "."
    save_inter_result = True

    print(" ********** preprocess test **********")
    eval_examples, eval_features = get_data(vocab_file, predict_file)
    print(" ******* run bert *******")
    res = inference_model(eval_features, np.int32, output_path, save_inter_result)
    print(" ********** postprocess **********")
    postprocess(eval_examples, eval_features, res, output_path)

