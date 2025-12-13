# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "faiss-cpu",
#     "sentence-transformers",
#     "numpy",
#     "tqdm",
#     "scikit-learn",
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
from sklearn.cluster import MiniBatchKMeans

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
    
    print(f"Total sentences found: {len(sentences)}")
    if len(sentences) > max_sentences:
        print(f"Truncating knowledge base to {max_sentences} sentences.")
        sentences = sentences[:max_sentences]
    
    sqrt_num = int(np.sqrt(len(sentences)))
    perfect_num = sqrt_num * sqrt_num
    if perfect_num != len(sentences):
        print(f"Adjusting to perfect square: {len(sentences)} -> {perfect_num}")
        sentences = sentences[:perfect_num]
        
    print(f"Loaded {len(sentences)} sentences.")
    return sentences

def load_queries_from_conversations(file_path):
    print(f"Loading queries from conversations file: {file_path}...")
    queries = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'conversations' in item:
                for conv in item['conversations']:
                    if conv.get('role') == 'user':
                        queries.append(conv.get('content', ''))
                        break
            else:
                raise ValueError(f"Expected 'conversations' format, got keys: {item.keys()}")
    print(f"Loaded {len(queries)} queries.")
    return queries

def train_kmeans(data, k, batch_size=10000):
    print(f"Training K-Means (K={k})...")
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, n_init='auto', random_state=42)
    kmeans.fit(data)
    return kmeans

def main():
    parser = argparse.ArgumentParser(description="Convert conversations format to labeled query format for router training")
    parser.add_argument("--conversations_path", type=str, default="data/train.jsonl", help="Path to conversations JSONL")
    parser.add_argument("--kb_path", type=str, default="data/knowledge_base/sentence_trex_data.json", help="Path to knowledge base JSON")
    parser.add_argument("--output_path", type=str, default="data/train_labeled.jsonl", help="Path to output JSONL")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5", help="Embedding model name")
    parser.add_argument("--top_k", type=int, default=32, help="Top K retrieval")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--max_sentences", type=int, default=1024*1024, help="Max sentences to load from KB")

    args = parser.parse_args()

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

    sentences = load_knowledge_base(args.kb_path, args.max_sentences)
    queries = load_queries_from_conversations(args.conversations_path)

    print(f"Loading embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)
    
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model embedding dimension: {embedding_dim}")

    print("Encoding knowledge base...")
    kb_embeddings = model.encode(sentences, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    
    print("Performing Residual Quantization...")
    
    num_items = len(kb_embeddings)
    num_clusters = int(np.sqrt(num_items))
    
    print("Step 1: Coarse Clustering (Row Keys)...")
    kmeans_coarse = train_kmeans(kb_embeddings, num_clusters)
    row_keys = kmeans_coarse.cluster_centers_
    row_labels = kmeans_coarse.labels_
    
    residuals = kb_embeddings - row_keys[row_labels]
    
    print("Step 2: Fine Clustering on Residuals (Col Keys)...")
    kmeans_fine = train_kmeans(residuals, num_clusters)
    col_keys = kmeans_fine.cluster_centers_
    col_labels = kmeans_fine.labels_
    
    print("Mapping original indices to Grid Indices...")
    grid_indices_map = row_labels * num_clusters + col_labels
    
    keys_path = os.path.join(os.path.dirname(args.output_path), "keys.pt")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    keys_tensor = torch.stack([torch.tensor(row_keys), torch.tensor(col_keys)], dim=0).float()
    torch.save(keys_tensor, keys_path)
    print(f"Saved semantic keys to {keys_path} with shape {keys_tensor.shape}")
    
    print("Building Faiss index...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(kb_embeddings)
    
    print("Encoding queries...")
    query_embeddings = model.encode(queries, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

    print(f"Searching Top-{args.top_k}...")
    all_D = []
    all_I = []
    
    for start in tqdm(range(0, len(query_embeddings), args.batch_size*8), desc="FAISS Searching"):
        end = start + args.batch_size*8
        batch = query_embeddings[start:end].astype(np.float32)
        D, I = index.search(batch, args.top_k)
        all_D.append(D)
        all_I.append(I)

    D = np.vstack(all_D)
    I = np.vstack(all_I)

    print(f"Saving results to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        for i, query in enumerate(queries):
            original_indices = I[i]
            mapped_indices = [int(grid_indices_map[idx]) for idx in original_indices]
            
            record = {
                "query": query,
                "target_indices": mapped_indices,
                "target_scores": D[i].tolist()
            }
            f.write(json.dumps(record) + "\n")
    
    meta_path = os.path.join(os.path.dirname(args.output_path), "meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "knowledge_num": len(sentences),
            "embedding_dim": embedding_dim * 2,
            "keys_path": keys_path
        }, f)
    print(f"Saved metadata to {meta_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()

