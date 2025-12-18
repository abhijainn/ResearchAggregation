#!/usr/bin/env python3
"""
Convert parquet embeddings into FAISS index + aligned metadata id file.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.linalg import norm
import faiss


# ================================
# CONFIG – update for your system
# ================================
PARQUET_PATH = r"C:\Users\sophi\Documents\research\FDL research\ResearchAggregation\data2\specter2_embeddings_all.parquet"
META_PATH    = r"C:\Users\sophi\Documents\research\FDL research\ResearchAggregation\data2\enhanced_final_meta_dataset_with_arxiv_deduped.csv"

# Write outputs into the repository data2 directory so the app can read them
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data2")

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "specter2.faiss")
IDS_PATH         = os.path.join(DATA_DIR, "specter2_ids.npy")
META_OUT_PATH    = os.path.join(DATA_DIR, "specter2_meta.parquet")


# ================================
# LOAD PARQUET + METADATA
# ================================
print("Loading parquet embedding file...")
table = pq.read_table(PARQUET_PATH)
df_emb = table.to_pandas()

# force consistent ID formatting
df_emb["source_id_clean"] = df_emb["source_id_clean"].astype(str).str.strip()

# extract embeddings column → numpy matrix [N, D]
print("Converting embedding column to NumPy...")
E = np.vstack(
    df_emb["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
)

# normalize once for cosine
print("Normalizing embeddings...")
E /= (norm(E, axis=1, keepdims=True) + 1e-12)

# load metadata CSV
print("Loading metadata CSV...")
meta = pd.read_csv(META_PATH)
meta["source_id_clean"] = meta["source_id_clean"].astype(str).str.strip()

# merge aligned properly
print("Merging metadata with embedding rows...")
merged = df_emb.merge(meta, on="source_id_clean", how="left")
merged = merged.drop(columns=["source_id_x"], errors="ignore")
merged = merged.drop(columns=["source_id_y"], errors="ignore")


# ================================
# BUILD FAISS INDEX
# ================================
d = E.shape[1]
print(f"Building FAISS index dimension={d} rows={E.shape[0]}...")
index = faiss.IndexFlatIP(d)    # cosine == dot because normalized
index.add(E)

print("Saving FAISS index...")
faiss.write_index(index, FAISS_INDEX_PATH)

print("Saving ID mapping...")
np.save(IDS_PATH, np.array(merged["source_id_clean"].tolist()))

print("Saving merged metadata parquet (optional)...")
merged.to_parquet(META_OUT_PATH, index=False)

print("FAISS index build complete.")
