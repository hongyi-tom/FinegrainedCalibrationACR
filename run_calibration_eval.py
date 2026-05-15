import os
import re
import sys
import warnings
import pandas as pd
import utils.calibrate_utils as u
from transformers import AutoTokenizer

# Suppress Warnings
sys.stderr = open(os.devnull, 'w')
warnings.filterwarnings("ignore")

def main():
    # Read, load, and filter generated code revisions
    result = sys.argv[1]
    correctness  = sys.argv[2]
    confidence = sys.argv[3]
    local_or_global = sys.argv[4]

    if re.search("Qwen", result):
        dir = "Qwen/"
    elif re.search("Llama", result):
        dir = "meta-llama/"
    elif re.search("deepseek", result):
        dir = "deepseek-ai/"
    elif re.search("DeepSeek", result):
        dir = "deepseek-ai/"

    benchmark = result.split("_")[-2] + "_" + result.split("_")[-1]
    test_generated = pd.read_json(benchmark + "/" + result + ".jsonl", lines=True)
    test_generated = u.calculate_confidence(result, benchmark, test_generated)

    if benchmark == "deepcode_bug" or benchmark == "deepcode_vul":
        train_generated = pd.read_json('deepcode_train/' + result.split('_')[0] + '_deepcode_train.jsonl', lines=True)
        train_generated = u.calculate_confidence(result.split('_')[0] + '_deepcode_train', 'deepcode_train', train_generated)

        if benchmark == "deepcode_bug":
            task_training_generated = train_generated.loc[train_generated.Type=='bug']
        elif benchmark == "deepcode_vul":
            task_training_generated = train_generated.loc[train_generated.Type=='vul']

    if benchmark == "codereviewqa_trans":
        task_training_generated = pd.read_json('codereviewqa_train/' + result.split('_')[0] + '_codereviewqa_train.jsonl', lines=True)
        task_training_generated = u.calculate_confidence(result.split('_')[0] + '_codereviewqa_train', 'codereviewqa_train', task_training_generated)

    # Run selected calibration method
    if local_or_global == "global":
        u.global_platt_scale(task_training_generated, test_generated, correctness, confidence)
    else:
        u.grid_search_local_platt_scale(task_training_generated, test_generated, correctness, confidence)

if __name__ == "__main__":
    main()