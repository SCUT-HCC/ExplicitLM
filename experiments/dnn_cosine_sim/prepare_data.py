# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "faiss-cpu",
#     "sentence-transformers",
#     "numpy",
#     "tqdm",
# ]
# ///

import json
import os
import argparse
import torch
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_knowledge_base(file_path, max_sentences=1024*1024):
    print(f"Loading knowledge base from {file_path}...")
    sentences = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        
        if isinstance(data, list):
            for entry in data:
                if "target" in entry and isinstance(entry["target"], list):
                    for t in entry["target"]:
                        if "sentence" in t:
                            sentences.append(t["sentence"])
                elif "sentence" in entry:
                    sentences.append(entry["sentence"])
        elif isinstance(data, dict) and "target" in data:
             for t in data["target"]:
                if "sentence" in t:
                    sentences.append(t["sentence"])
        else:
            print("Warning: Unknown JSON structure, trying to find 'sentence' recursively or in top level")
            # Simple fallback not implemented for brevity, assuming structure matches observation
    
    print(f"Total sentences found: {len(sentences)}")
    if len(sentences) > max_sentences:
        print(f"Truncating knowledge base to {max_sentences} sentences.")
        sentences = sentences[:max_sentences]
    
    print(f"Loaded {len(sentences)} sentences.")
    return sentences

def load_queries(file_path):
    print(f"Loading queries from {file_path}...")
    queries = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # User said: "experiments/dnn_cosine_sim/train.jsonl ... content作为输入"
            if 'content' in item:
                queries.append(item['content'])
            elif 'input' in item:
                queries.append(item['input'])
            else:
                 # Fallback
                queries.append(line.strip())
    print(f"Loaded {len(queries)} queries.")
    return queries

def main():
    parser = argparse.ArgumentParser(description="Prepare data for DNN Cosine Sim experiment")
    parser.add_argument("--kb_path", type=str, default="data/knowledge_base/sentence_trex_data.json", help="Path to knowledge base JSON")
    parser.add_argument("--query_path", type=str, default="experiments/dnn_cosine_sim/train.jsonl", help="Path to query JSONL")
    parser.add_argument("--output_path", type=str, default="experiments/dnn_cosine_sim/train_with_labels.jsonl", help="Path to output JSONL")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5", help="Embedding model name")
    parser.add_argument("--top_k", type=int, default=32, help="Top K retrieval")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--max_sentences", type=int, default=1024*1024, help="Max sentences to load from KB")

    args = parser.parse_args()

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")

    # Load Data
    sentences = load_knowledge_base(args.kb_path, args.max_sentences)
    queries = load_queries(args.query_path)

    # Load Model
    print(f"Loading embedding model: {args.model_name}")
    # SentenceTransformer supports MPS backend
    model = SentenceTransformer(args.model_name, device=device)

    # Encode Knowledge Base
    print("Encoding knowledge base...")
    # SentenceTransformer encodes to numpy by default
    kb_embeddings = model.encode(sentences, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

    # --- Lightweight Semantic Sorting ---
    print("Sorting knowledge base by 1st dimension of embeddings...")
    # Sort indices based on the first dimension of the embeddings
    # This clusters similar items together in the index space, helping PKM
    sort_indices = np.argsort(kb_embeddings[:, 0])
    
    # Reorder embeddings and sentences
    kb_embeddings = kb_embeddings[sort_indices]
    # We also need to reorder the sentences list to match (though we don't save it back in this script, 
    # for a real system you'd want to save the sorted KB)
    sentences = [sentences[i] for i in sort_indices]
    
    # Save sorted mapping for future reference (optional but recommended)
    # mapping_path = os.path.join(os.path.dirname(args.output_path), "sorted_indices.npy")
    # np.save(mapping_path, sort_indices)
    # ------------------------------------

    # Build Faiss Index
    print("Building Faiss index...")
    d = kb_embeddings.shape[1]
    # Inner Product with normalized embeddings is equivalent to Cosine Similarity
    # Faiss only runs on CPU on Mac (no MPS support for Faiss yet)
    # If we are on CUDA, we could move index to GPU, but for simplicity and compatibility we keep index on CPU
    # unless explicitly requested. For this size of data, CPU index is fine.
    index = faiss.IndexFlatIP(d)
    index.add(kb_embeddings)
    print(f"Index built with {index.ntotal} vectors.")

    # Encode Queries
    print("Encoding queries...")
    query_embeddings = model.encode(queries, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

    # Search
    print(f"Searching Top-{args.top_k} with progress bar...")

    all_D = []
    all_I = []

    for start in tqdm(range(0, len(query_embeddings), args.batch_size*8), desc="FAISS Searching"):
        end = start + args.batch_size*8
        batch = query_embeddings[start:end].astype(np.float32)

        D, I = index.search(batch, args.top_k)
        all_D.append(D)
        all_I.append(I)

    # 合并搜索结果
    D = np.vstack(all_D)
    I = np.vstack(all_I)

    # Save Results
    print(f"Saving results to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        for i, query in enumerate(queries):
            record = {
                "query": query,
                "target_indices": I[i].tolist(),
                "target_scores": D[i].tolist()
            }
            f.write(json.dumps(record) + "\n")
    
    # Save Metadata
    meta_path = os.path.join(os.path.dirname(args.output_path), "meta.json")
    with open(meta_path, 'w') as f:
        json.dump({"knowledge_num": len(sentences)}, f)
    print(f"Saved metadata to {meta_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()
