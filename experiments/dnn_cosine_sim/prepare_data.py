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
    
    # Ensure we have exactly a perfect square number of sentences for the grid
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

def train_kmeans(data, k, batch_size=10000):
    print(f"Training K-Means (K={k})...")
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, n_init='auto', random_state=42)
    kmeans.fit(data)
    return kmeans

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
    
    # --- Residual Quantization (Coarse-to-Fine) ---
    print("Performing Residual Quantization...")
    
    num_items = len(kb_embeddings)
    num_clusters = int(np.sqrt(num_items)) # 1024
    
    # 1. Coarse Quantization (Row Keys)
    print("Step 1: Coarse Clustering (Row Keys)...")
    kmeans_coarse = train_kmeans(kb_embeddings, num_clusters)
    row_keys = kmeans_coarse.cluster_centers_ # [1024, 768]
    row_labels = kmeans_coarse.labels_ # [N]
    
    # Calculate Residuals
    # Residual = Original - Coarse_Centroid
    residuals = kb_embeddings - row_keys[row_labels]
    
    # 2. Fine Quantization (Col Keys)
    print("Step 2: Fine Clustering on Residuals (Col Keys)...")
    kmeans_fine = train_kmeans(residuals, num_clusters)
    col_keys = kmeans_fine.cluster_centers_ # [1024, 768]
    col_labels = kmeans_fine.labels_ # [N]
    
    # 3. Reorder Data based on (Row, Col)
    # We need to sort the data so that it matches the grid structure:
    # Index = Row_Label * 1024 + Col_Label
    # Wait, the grid structure implies that for every Row_Label, there are exactly 1024 Col_Labels.
    # But K-Means doesn't guarantee balanced clusters!
    # This is a problem. Standard PKM assumes a dense grid.
    # If we use K-Means, some (u, v) slots might be empty, and some might have multiple items.
    # BUT, our model predicts (u, v).
    # If we map each item to its (u, v), we get a target index.
    # The "knowledge_num" in the model implies a fixed address space.
    # We need to map the K-Means result to the linear index [0, N-1].
    # Actually, we can just store the (u, v) for each item.
    # But the FAISS index retrieval returns an integer ID.
    # We need to align the FAISS ID with the (u, v) structure.
    # Strategy:
    # We will NOT reorder the sentences physically to form a perfect grid (impossible with K-Means).
    # Instead, we will keep the sentences as is (or sorted by ID), 
    # BUT we need to tell the training script what the target (u, v) is for each item.
    # Currently, `train_router.py` assumes `target_indices` are linear indices in [0, N-1], 
    # and `MemoryGate` decomposes them: u = idx // 1024, v = idx % 1024.
    # This decomposition enforces the grid structure.
    # IF we want to use K-Means (u, v), we have a mismatch:
    # Item K has cluster (u_k, v_k).
    # Its linear index in the grid would be u_k * 1024 + v_k.
    # BUT multiple items might map to the same (u_k, v_k)! (Collision)
    # And some (u, v) might be empty.
    # This is fine for training! The model just learns to predict (u_k, v_k) for that item.
    # The "Collision" just means multiple items share the same memory slot.
    # The "Empty" just means some slots are unused.
    # So, we need to convert the FAISS retrieval results (which are indices into `sentences`)
    # into the Grid Indices (u * 1024 + v).
    
    # So:
    # 1. We don't need to sort `sentences` or `kb_embeddings`.
    # 2. We build FAISS index on `kb_embeddings` (original order).
    # 3. When we retrieve Top-K, we get indices `I` (original indices).
    # 4. We need a mapping: Original_Index -> Grid_Index.
    #    Grid_Index = row_labels[Original_Index] * 1024 + col_labels[Original_Index].
    # 5. We replace the retrieved indices `I` with these `Grid_Indices` in the output JSONL.
    
    print("Mapping original indices to Grid Indices...")
    grid_indices_map = row_labels * num_clusters + col_labels
    
    # Save Keys
    # Keys shape: [2, 1024, 768]
    keys_path = os.path.join(os.path.dirname(args.output_path), "keys.pt")
    keys_tensor = torch.stack([torch.tensor(row_keys), torch.tensor(col_keys)], dim=0).float()
    torch.save(keys_tensor, keys_path)
    print(f"Saved semantic keys to {keys_path} with shape {keys_tensor.shape}")
    
    # Build Faiss Index (on original embeddings)
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
            # Map retrieved original indices to Grid Indices
            original_indices = I[i]
            mapped_indices = [int(grid_indices_map[idx]) for idx in original_indices]
            
            record = {
                "query": query,
                "target_indices": mapped_indices, # These are now (u*1024 + v)
                "target_scores": D[i].tolist()
            }
            f.write(json.dumps(record) + "\n")
    
    # Save Metadata
    # Note: embedding_dim is set to 768 * 2 = 1536 so that MemoryGate splits it into 768+768
    meta_path = os.path.join(os.path.dirname(args.output_path), "meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "knowledge_num": len(sentences), # This is just for config, actual addressing is 1024*1024
            "embedding_dim": embedding_dim * 2, # 1536
            "keys_path": keys_path
        }, f)
    print(f"Saved metadata to {meta_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()
