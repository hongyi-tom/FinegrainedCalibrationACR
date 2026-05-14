import re
import gc
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def attention_rollout(attentions, normalise_causal=False):
    """
    Perform attention rollout for a causal language model.
    Parameters:
        attentions: tuple of attention tensors, each of shape (batch_size, num_heads, seq_len, seq_len)
        normalise_causal: bool, whether to normalise by receptive field size (recommended for causal/decoder models)
    Returns:
        rollout: tensor of shape (batch_size, seq_len, seq_len) representing the cumulative attention from each token to every other token
    """
    # Move all attentions to CPU immediately
    attentions = tuple(attn.cpu() for attn in attentions)
    batch_size, num_heads, seq_len, _ = attentions[0].shape
    
    # Average attention heads for each layer
    # Shape: (num_layers, batch_size, seq_len, seq_len)
    layer_attentions = torch.stack([
        attn.mean(dim=1) for attn in attentions
    ])
    
    if normalise_causal:
        # Normalise by receptive field size (number of tokens each position can attend to)
        # For causal masking: position i can attend to positions 0...i (i+1 tokens total)
        receptive_field = torch.arange(1, seq_len + 1)  # CPU
        receptive_field = receptive_field.view(1, 1, seq_len, 1)
        
        # Normalise each row by its receptive field size
        layer_attentions = layer_attentions / receptive_field
    
    # Add residual connections (identity matrix)
    # This accounts for the residual pathway in transformers
    eye = torch.eye(seq_len)  
    eye = eye.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)
    
    # Blend: 0.5 * attention + 0.5 * identity
    layer_attentions = 0.5 * layer_attentions + 0.5 * eye
    
    # Re-normalise so each row sums to 1
    layer_attentions = layer_attentions / layer_attentions.sum(dim=-1, keepdim=True)
    
    # Perform rollout: multiply attention matrices from all layers
    rollout = layer_attentions[0]  # Start with first layer
    
    for i in range(1, len(attentions)):
        # Matrix multiplication to propagate attention
        # rollout @ layer_attentions[i] gives us the cumulative attention
        rollout = torch.matmul(rollout, layer_attentions[i])
    
    # Clean up before returning
    del layer_attentions

    return rollout


def main():
    # Read and load inference results from command-line arguments
    result = sys.argv[1]
    benchmark = result.split("_")[-2] + "_" + result.split("_")[-1]
    generated = pd.read_json(benchmark + "/" + result + ".jsonl", lines=True)

    # Read and load relevant benchmark
    if result.split("_")[-1] == 'train':
        dataset = pd.read_parquet(benchmark + "/" + benchmark + ".parquet").reset_index()
    else:
        dataset = pd.read_parquet(benchmark + "/" + benchmark + "_test" + ".parquet").reset_index()

    # Load relevant model
    if re.search("Qwen", result):
        dir = "Qwen/"
    elif re.search("Llama", result):
        dir = "meta-llama/"
    elif re.search("deepseek", result):
        dir = "deepseek-ai/"
    elif re.search("DeepSeek", result):
        dir = "deepseek-ai/"

    model_name = dir + result.split("_")[0]
    t = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",
                                            attn_implementation = "eager",
                                            offload_folder="offload_weights",  # Disk offload 
                                            offload_state_dict=True)  # Reduce load-time memory

    # Make sure the model outputs attentions
    m.config.output_attentions = True
    m.config.use_cache = False

    # Construct full sequences
    full_sequences = dataset.prompts + generated.generated_sequence_f
    full_sequences = full_sequences.to_list()

    # Calculate attention rollouts
    row = 0
    store_rollout = []
    failed_indices = []
    for seq in tqdm(full_sequences):
        try:
            len_out = len(generated.iloc[row].token_ids_f)
            
            # Skips situation where there was no generated sequence
            if len_out == 0:
                store_rollout.append([])
                continue
            
            # Extract attention scores and offload to CPU
            inputs = t(seq, return_tensors="pt")
            with torch.no_grad():
                outputs = m(input_ids=inputs["input_ids"])
                attentions_cpu = tuple(attn.cpu() for attn in outputs.attentions)

            # Delete GPU tensors to prevent OOM
            del outputs
            del inputs
            gc.collect()

            # Calculate attention rollout
            rollout = attention_rollout(attentions_cpu)
            rollout = rollout[0].numpy()

            temp = []
            for i in range(rollout.shape[1]):
                temp.append(np.sum(rollout[:, i][i:]))
                
            store_rollout.append(temp[-len_out:])
            
        except:
            # Catch sequence that causes OOM
            print(f"\nOOM error at index {row}")
            failed_indices.append(row)
            store_rollout.append(None) 
        
        finally:
            row += 1

    print(f"\nProcessing complete. Failed indices: {failed_indices}")

    # Fill OOM sequences with placeholder attention rollouts
    for index in failed_indices:
        len_seq = len(generated.iloc[index].token_ids_f)
        filler = [0.9] * len_seq 
        store_rollout[index] = filler

    # Save Results
    generated['attention_mass_key'] = store_rollout
    generated.to_json(benchmark + "/" + result + ".jsonl", orient='records', lines=True)

    # Conduct sanity check
    print("### SANITY CHECK ###")
    for n in range(len(generated)):
        example = generated.iloc[n]
        if len(example.softmax_probabilities_f) != len(example.attention_mass_key):
            print(n)
            print(len(example.softmax_probabilities_f), len(example.attention_mass_key))

if __name__ == "__main__":
    main()