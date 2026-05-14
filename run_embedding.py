import sys
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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

    # Load embedding model
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

    # Extract embeddings of full sequences
    embeddings_set = []
    for i in tqdm(range(len(dataset))):
        embeddings = model.encode(dataset.prompts[i] + generated.generated_sequence_f[i]).tolist()
        embeddings_set.append(embeddings)
    
    # Save Results
    generated['embeddings'] = embeddings_set
    generated.to_json(benchmark + "/" + result + ".jsonl", orient='records', lines=True)

if __name__ == "__main__":
    main()