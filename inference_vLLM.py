import sys
import ast
import math
import torch
import pandas as pd
import data_processor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def get_tensor_parallel_size():
    """
    Determines the tensor parallel size based on available CUDA devices.
    Returns:
        int: The number of available CUDA GPUs if CUDA is enabled, otherwise 1 (defaulting to single-device execution).
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1

def truncate_prompt(prompts, model_name, max_tokens=4000):
    """
    Truncates the list of prompts so that each does not exceed a specified maximum number of tokens for a given model.
    Args:
        prompts (List[str]): A list of input prompt strings.
        model_name (str): HuggingFace model name.
        max_tokens (int, optional): The maximum number of tokens allowed per prompt. Defaults to 4000.
    Returns:
        List[str]: A list of prompts with a maximum token length. 
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    truncated_prompts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        
        if len(tokens) <= max_tokens:
            truncated_prompts.append(prompt)
        else:
            truncated_tokens = tokens[:max_tokens]
            truncated_prompts.append(tokenizer.decode(truncated_tokens, skip_special_tokens=True))
    
    return truncated_prompts
 
def main():
    # Read model name from command-line arguments
    model_name = sys.argv[1]

    # Read and load benchmark from command-line arguments
    benchmark = sys.argv[2]
    if benchmark == "deepcode_bug":
        data = pd.read_parquet("deepcode_bug/deepcode_bug_test.parquet")
        prompt_set = data_processor.process_deepcode(data)
        post_revision = data.post_reduced

    elif benchmark == "deepcode_vul":
        data = pd.read_parquet("deepcode_vul/deepcode_vul_test.parquet")
        prompt_set = data_processor.process_deepcode(data)
        post_revision = data.post_reduced

    elif benchmark == "codereviewqa_trans":
        data = pd.read_parquet("codereviewqa_trans/codereviewqa_trans_test.parquet")
        prompt_set = data_processor.process_codereviewqa(data)
        post_revision = data.post_review_code

    elif benchmark == "codereviewqa_train":
        data = pd.read_parquet("codereviewqa_train/codereviewqa_train.parquet")
        prompt_set = data_processor.process_codereviewqa(data)
        post_revision = data.post_review_code

    elif benchmark == "deepcode_train":
        data = pd.read_parquet("deepcode_train/deepcode_train.parquet")
        prompt_set = data_processor.process_deepcode(data)
        post_revision = data.post_reduced
    
    # Estimate maximum output length based on length of ground-truth post revision code snippet 
    estimate_output_len = data_processor.estimate_max_length(sys.argv[1], post_revision)

    # Free large intermediate objects from memory
    del data
    del post_revision
    
    # Load LLM with tensor parallelism and native bfloat16 precision
    model = LLM(model=model_name, dtype="bfloat16", tensor_parallel_size=get_tensor_parallel_size(), trust_remote_code=True)
    
    # Configure deterministic sampling (temperature=0) and request token-level logprobs
    sampling_params = SamplingParams(max_tokens=estimate_output_len, temperature=0, logprobs=1, n=1, 
                                        stop=['[_END_FIXED_CODE]'], include_stop_str_in_output=True)
    
    # Run Inference
    results = []
    for example in model.generate(truncate_prompt(prompt_set, model_name), sampling_params=sampling_params):
        output = example.outputs[0]

        # vLLM gives access to the token-level info in `output.sequences`
        token_ids = output.token_ids  # list of token IDs
        generated_sequence = output.text  # generated text
        softmax_probabilities = [math.exp(list(d.values())[0].logprob) for d in output.logprobs]  # convert logprobs to probs
        
        results.append({
            "token_ids": token_ids,
            "generated_sequence": generated_sequence,
            "softmax_probabilities": softmax_probabilities
    })

    # Save Results
    print("\n=== Saving Results ===")
    output_path = "{benchmark}/{model}_{benchmark}".format(
        model=model_name.split("/")[1], 
        benchmark=benchmark
    )

    df = pd.DataFrame(results)
    df.to_json(output_path + ".jsonl", orient="records", lines=True)
    print(f"Results saved to {output_path}.jsonl")

if __name__ == "__main__":
    main()