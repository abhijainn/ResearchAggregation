from datasets import load_dataset
import torch, numpy as np
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

MODEL_NAME = "allenai/specter2_base"
ADAPTER = "allenai/specter2"
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoAdapterModel.from_pretrained(MODEL_NAME)
# SPECTER2 is trained for retrieval; use base adapter
model.load_adapter(ADAPTER, source="hf", load_as="specter2")
model.set_active_adapters("specter2")
# gen embeddings in eval mode, e.g., no dropout; attempt to use GPU if available
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 51.7k papers in the set
ds = load_dataset("csv", data_files={"train": "arxiv_metadata.csv"})["train"]

def concat_title_abs(batch):
    texts = []
    for t, a in zip(batch["title"], batch["abstract"]):
        t = t or ""
        a = a or ""
        texts.append(f"{t} {tok.sep_token} {a}")
    return {"_text": texts}

ds = ds.map(concat_title_abs, batched=True, remove_columns=[])

def encode_batch(batch):
    with torch.inference_mode():
        enc = tok(batch["_text"], padding="longest", truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state[:, 0]
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return {"embedding": out.cpu().numpy().astype("float32")}

ds = ds.map(encode_batch, batched=True, batch_size=256)
ds = ds.remove_columns(["_text"])
# Output: 768-dim embedding
ds.to_parquet("arxiv_specter2_embeddings.parquet")  
