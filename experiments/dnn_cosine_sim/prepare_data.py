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
from sklearn.decomposition import PCA

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
    
    # Ensure we have exactly a perfect square number of sentences for the grid
    # If not, we might need to pad or truncate further.
    # The model expects knowledge_num to be square.
    sqrt_num = int(np.sqrt(len(sentences)))
    perfect_num = sqrt_num * sqrt_num
    if perfect_num != len(sentences):
        print(f"Adjusting to perfect square: {len(sentences)} -> {perfect_num}")
        sentences = sentences[:perfect_num]
        
    print(f"Loaded {len(sentences)} sentences.")
    return sentences

def load_queries(file_path):
    print(f"Loading queries from {file_path}...")
    queries = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'content' in item:
                queries.append(item['content'])
            elif 'input' in item:
                queries.append(item['input'])
            else:
                queries.append(line.strip())
    print(f"Loaded {len(queries)} queries.")
    return queries

def compute_pca_score(embeddings):
    """Compute PC1 projection score for sorting"""
    pca = PCA(n_components=1)
    # Fit on a subset if data is too large for speed, but 1M is fine for PCA
    if len(embeddings) > 100000:
        pca.fit(embeddings[:100000])
    else:
        pca.fit(embeddings)
    scores = pca.transform(embeddings).flatten()
    return scores

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
    model = SentenceTransformer(args.model_name, device=device)
    
    # Check dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model embedding dimension: {embedding_dim}")
    if embedding_dim != 768:
        print("Warning: Expected 768 dimension for this experiment setup.")

    # Encode Knowledge Base
    print("Encoding knowledge base...")
    kb_embeddings = model.encode(sentences, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    
    # --- Hierarchical PCA Sorting & Key Generation ---
    print("Performing Hierarchical PCA Sorting...")
    
    num_items = len(kb_embeddings)
    num_rows = int(np.sqrt(num_items))
    half_dim = embedding_dim // 2
    
    # 1. Global Sort by Row Subspace (First Half)
    print("Sorting rows...")
    sub1 = kb_embeddings[:, :half_dim]
    row_scores = compute_pca_score(sub1)
    row_sort_indices = np.argsort(row_scores)
    
    kb_embeddings = kb_embeddings[row_sort_indices]
    sentences = [sentences[i] for i in row_sort_indices]
    
    # 2. Local Sort by Col Subspace (Second Half)
    print("Sorting columns within rows...")
    # Reshape to [rows, cols, dim]
    kb_reshaped = kb_embeddings.reshape(num_rows, num_rows, embedding_dim)
    
    # We need to sort each row independently based on sub2
    sorted_rows = []
    final_indices_map = np.zeros(num_items, dtype=int) # Just for tracking if needed
    
    # To compute Col Keys, we need the sorted data
    # We will do the sort in place
    for i in tqdm(range(num_rows), desc="Sorting Rows"):
        row_data = kb_reshaped[i] # [cols, dim]
        sub2 = row_data[:, half_dim:]
        col_scores = compute_pca_score(sub2)
        col_sort_indices = np.argsort(col_scores)
        
        # Reorder this row
        kb_reshaped[i] = row_data[col_sort_indices]
        
        # Also reorder sentences
        # Calculate global indices
        start_idx = i * num_rows
        current_sentences_slice = sentences[start_idx : start_idx + num_rows]
        sentences[start_idx : start_idx + num_rows] = [current_sentences_slice[j] for j in col_sort_indices]

    # Flatten back
    kb_embeddings = kb_reshaped.reshape(num_items, embedding_dim)
    
    # 3. Compute Keys (Centroids)
    print("Computing Semantic Keys...")
    
    # Row Keys: Average of Sub1 for each row
    # [num_rows, num_cols, dim] -> mean over cols -> [num_rows, dim] -> take sub1
    row_keys = np.mean(kb_reshaped, axis=1)[:, :half_dim] # [1024, 384]
    
    # Col Keys: Average of Sub2 for each column (across all rows)
    # [num_rows, num_cols, dim] -> mean over rows -> [num_cols, dim] -> take sub2
    col_keys = np.mean(kb_reshaped, axis=0)[:, half_dim:] # [1024, 384]
    
    # Save Keys
    keys_path = os.path.join(os.path.dirname(args.output_path), "keys.pt")
    keys_tensor = torch.stack([torch.tensor(row_keys), torch.tensor(col_keys)], dim=0).float()
    torch.save(keys_tensor, keys_path)
    print(f"Saved semantic keys to {keys_path} with shape {keys_tensor.shape}")
    
    # Save Sorted Indices (Optional, but good for debug)
    # ------------------------------------

    # Build Faiss Index with SORTED embeddings
    print("Building Faiss index...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(kb_embeddings)
    
    # Encode Queries
    print("Encoding queries...")
    query_embeddings = model.encode(queries, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

    # Search
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
        json.dump({
            "knowledge_num": len(sentences),
            "embedding_dim": embedding_dim,
            "keys_path": keys_path
        }, f)
    print(f"Saved metadata to {meta_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()
