import ast
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def deepcode_prompt(rule_key, rule_message, pre_code):
    """
    Constructs the inference prompt for samples from DeepCode AI Fix.
    Parameters:
        rule_key (str): Identifier of the static analysis rule violation.
        rule_message (str): Description of the static analysis rule violation.
        pre_code (str): The original buggy code snippet.
    Returns:
        str: A formatted prompt string that will instruct the LLM to conduct automated program/vulnerability repair.
    """
    prompt = f"Fix {rule_key}\n{rule_message}\n[_BEGIN_BUGGY_CODE]\n{pre_code}\n[_END_BUGGY_CODE]\nPlease return only fixed code, wrapped in the special markers.\n[_BEGIN_FIXED_CODE]\n"
    return prompt


def process_deepcode(benchmark):
    """
    Applies inference prompt across all samples for any DeepCode AI Fix dataset.
    Parameters:
        benchmark (pandas.DataFrame): Dataset from DeepCode AI Fix i.e., DCF-Bug, DCF-Vul, DCF-Train.
    Returns:
        list[str]: A list of formatted prompt strings for LLM-based automated program/vulnerability repair.
    """
    prompt_set = benchmark.apply(lambda row: deepcode_prompt(row.rule, row.message, row.pre_reduced), axis=1)
    return list(prompt_set)


def codereviewqa_prompt(code_review, pre_code):
    """
    Constructs the inference prompt for samples from CR-Trans.
    Parameters:
        code_review (str): The inline code review comment from a human reviewer.
        pre_code (str): The original code snippet under review.
    Returns:
        str: A formatted prompt string that will instruct the LLM to conduct automated code refinement.
    """
    prompt = f"The following code snippet has received a code review.\n[_BEGIN_BUGGY_CODE]\n{pre_code}\n[_END_BUGGY_CODE]\n[_BEGIN_CODE_REVIEW]\n{code_review}\n[_END_CODE_REVIEW]\nGenerate a revised version of the code snippet according to the code review.\nPlease return only fixed code, wrapped in the special markers.\n[_BEGIN_FIXED_CODE]\n"
    return prompt


def process_codereviewqa(benchmark):
    """
    Applies inference prompt across all samples for either the training or test set of CR-Trans.
    Parameters:
        benchmark (pandas.DataFrame): Dataset from CR-Trans i.e., training, test.
    Returns:
        list[str]: A list of formatted prompt strings for LLM-based automated code refinement.
    """
    prompt_set = benchmark.apply(lambda row: codereviewqa_prompt(row.code_review, row.pre_review_code), axis=1)
    return list(prompt_set)


def estimate_max_length(model, data):
    """
    Estimates a conservative maximum token generation cut off based on lengths of tokenised post-revision code snippets.
    Parameters:
        model (str): HuggingFace model name.
        data (List[str]): A list of ground truth post revision code snippets.
    Returns:
        int: Estimated maximum token generation cut off length.
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenized = [tokenizer.encode(text, add_special_tokens=True) for text in data]

    lengths = [len(tokens) for tokens in tokenized]
    mean = np.mean(lengths)
    std = np.std(lengths)  
    estimate = round(mean + 3 * std)
    return estimate


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