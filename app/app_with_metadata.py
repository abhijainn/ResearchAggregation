import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from adapters import AutoAdapterModel

# -----------------------------
# CONFIG — edit paths if needed
# -----------------------------
SPECTER2_PARQUET = "arxiv_specter2_embeddings.parquet"  # must have 'embedding','title','abstract'; optional 'filepath','url','arxiv_id'
METADATA_CSV = "arxiv_metadata.csv"                     # optional; adds published/updated/doi/filename
TITLE_COL = "title"
ABSTR_COL = "abstract"
EMBED_COL = "embedding"
FILEPATH_COL = "filepath"
URL_COL = "url"
ARXIV_ID_COL = "arxiv_id"

TOPK_INITIAL = 50
TOPK_SHOW = 15

# Models
SPECTER2_MODEL = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"
RERANK_MODEL = "mixedbread-ai/mxbai-embed-large-v1"  # embedding reranker via cosine
LLM_GPU_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_CPU_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# -----------------------------
# Helpers
# -----------------------------
def _safe_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return np.ascontiguousarray(x / norms)

@st.cache_resource
def load_data_and_index(parquet_path: str, metadata_csv: str):
    if not Path(parquet_path).exists():
        st.error(f"Parquet not found: {parquet_path}")
        st.stop()
    df = pd.read_parquet(parquet_path)

    # Ensure text columns
    for col in (TITLE_COL, ABSTR_COL):
        if col not in df.columns:
            df[col] = ""
    for col in (FILEPATH_COL, URL_COL, ARXIV_ID_COL):
        if col not in df.columns:
            df[col] = ""

    # Optional metadata enrichment
    meta = None
    if Path(metadata_csv).exists():
        meta = pd.read_csv(metadata_csv)
        # Keep only relevant columns if present
        keep = ["filepath","filename","arxiv_id","short_id","version","published","updated","doi","title","abstract"]
        cols = [c for c in keep if c in meta.columns]
        meta = meta[cols].copy()

        # Prefer merge on filepath if available; else try arxiv_id
        if FILEPATH_COL in df.columns and "filepath" in meta.columns:
            df = df.merge(meta, how="left", left_on=FILEPATH_COL, right_on="filepath", suffixes=("", "_meta"))
        elif ARXIV_ID_COL in df.columns and "arxiv_id" in meta.columns:
            df = df.merge(meta, how="left", left_on=ARXIV_ID_COL, right_on="arxiv_id", suffixes=("", "_meta"))
        # If title/abstract exist in both, keep primary df versions

        # Fill missing text fields from metadata if parquet lacked them
        if df[TITLE_COL].eq("").any() and "title_meta" in df.columns:
            df[TITLE_COL] = df[TITLE_COL].mask(df[TITLE_COL].eq(""), df["title_meta"].fillna(""))
        if df[ABSTR_COL].eq("").any() and "abstract_meta" in df.columns:
            df[ABSTR_COL] = df[ABSTR_COL].mask(df[ABSTR_COL].eq(""), df["abstract_meta"].fillna(""))

    # Embeddings
    if EMBED_COL not in df.columns:
        st.error(f"Parquet must contain an '{EMBED_COL}' column with arrays.")
        st.stop()
    X = np.vstack(df[EMBED_COL].to_numpy())
    X = _safe_norm(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    return df, index

@st.cache_resource
def load_specter2():
    tok = AutoTokenizer.from_pretrained(SPECTER2_MODEL, use_fast=True)
    model = AutoAdapterModel.from_pretrained(SPECTER2_MODEL)
    model.load_adapter(SPECTER2_ADAPTER, source="hf", load_as="specter2")
    model.set_active_adapters("specter2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tok, model, device

@torch.inference_mode()
def encode_query_specter2(tok, model, device, text: str) -> np.ndarray:
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    cls = model(**enc).last_hidden_state[:, 0]
    cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
    return cls[0].detach().cpu().numpy().astype("float32")[None, :]

@st.cache_resource
def load_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(RERANK_MODEL, device=device)

def rerank_with_mxbai(rer_model, query: str, titles: pd.Series, abstracts: pd.Series, batch_size: int = 64):
    texts = (titles.fillna("").astype(str) + " [SEP] " + abstracts.fillna("").astype(str)).tolist()
    qv = rer_model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    dv = rer_model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    scores = (dv @ qv[0].astype(np.float32))
    return scores

def trim(s: str, n=420):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"

def link_for_row(row):
    url = row.get(URL_COL, "") or ""
    if isinstance(url, str) and url:
        return url
    fp = row.get(FILEPATH_COL, "") or ""
    if isinstance(fp, str) and fp:
        return f"file://{Path(fp).absolute()}"
    return ""

def fmt_date(s):
    if pd.isna(s) or not isinstance(s, str) or not s:
        return ""
    # s may already be ISO; keep as-is but shorten if needed
    return s.split("T")[0] if "T" in s else s

# ---- LLM explainer ----
@st.cache_resource
def load_llm():
    use_gpu = torch.cuda.is_available()
    name = LLM_GPU_MODEL if use_gpu else LLM_CPU_MODEL
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        device_map="auto" if use_gpu else None
    )
    model.eval()
    return tok, model, name

def build_prompt(query: str, title: str, abstract: str) -> str:
    return (
        "You are an expert in NLP literature. "
        "Given a user query and a paper (title + abstract), explain briefly and precisely "
        "WHY this paper is relevant to the query. Mention concrete overlaps (task, method, dataset, findings). "
        "Avoid generic statements.\n\n"
        f"Query:\n{query}\n\n"
        f"Paper Title:\n{title}\n\n"
        f"Paper Abstract:\n{abstract}\n\n"
        "Answer (2–4 sentences):"
    )

@torch.inference_mode()
def explain_relevance(tok, model, query: str, title: str, abstract: str, max_new_tokens: int = 160) -> str:
    prompt = build_prompt(query, title or "", abstract or "")
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.2,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    if "Answer" in text:
        text = text.split("Answer")[-1].lstrip(":").strip()
    return text.strip()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Local NLP Paper Search", layout="wide")
st.title("🔎 Local NLP Paper Search")

with st.sidebar:
    st.header("Settings")
    topk = st.slider("Initial ANN top‑k", 10, 200, TOPK_INITIAL, 10)
    showk = st.slider("Show top‑k", 5, 50, TOPK_SHOW, 5)
    do_rerank = st.checkbox("Re‑rank with mxbai embeddings", value=True)
    gen_llm = st.checkbox("Generate ‘why relevant’ (open‑source LLM)", value=True)
    st.caption("Disable re‑ranking and LLM for faster results.")

query = st.text_input("Query (e.g., “Large language models for automatic speech recognition”):", value="")
search = st.button("Search", type="primary", use_container_width=True)

df, index = load_data_and_index(SPECTER2_PARQUET, METADATA_CSV)
tok_s2, model_s2, device_s2 = load_specter2()
rer_model = load_reranker() if do_rerank else None
tok_llm = model_llm = model_llm_name = None
if gen_llm:
    tok_llm, model_llm, model_llm_name = load_llm()

if search and query.strip():
    with st.spinner("Encoding query with SPECTER2 and searching ANN…"):
        q = encode_query_specter2(tok_s2, model_s2, device_s2, query.strip())
        D, I = index.search(q, topk)

    cand = df.iloc[I[0]].copy()
    cand["ann_cosine"] = D[0]

    if do_rerank:
        with st.spinner("Re‑ranking with mxbai embeddings…"):
            r_scores = rerank_with_mxbai(rer_model, query.strip(), cand[TITLE_COL], cand[ABSTR_COL])
            cand["rerank_cosine"] = r_scores
            cand = cand.sort_values("rerank_cosine", ascending=False)
    else:
        cand = cand.sort_values("ann_cosine", ascending=False)

    st.subheader("Results")
    for i, (_, row) in enumerate(cand.head(showk).iterrows(), start=1):
        title = row[TITLE_COL] or "(untitled)"
        abs_full = row[ABSTR_COL]
        ann_s = f"{row['ann_cosine']:.3f}"
        #link = link_for_row(row)

        st.markdown(f"### {i}. {title}")
        #if link:
        #    st.markdown(f"[Open]({link})")

        # Scores
        if "rerank_cosine" in row:
            st.caption(f"ANN cosine: {ann_s}  |  Re‑rank cosine: {row['rerank_cosine']:.3f}")
        else:
            st.caption(f"ANN cosine: {ann_s}")

        # LLM explanation or abstract snippet
        if gen_llm and tok_llm is not None:
            with st.spinner("Explaining relevance…"):
                try:
                    expl = explain_relevance(tok_llm, model_llm, query.strip(), title, abs_full)
                except Exception as e:
                    expl = f"(LLM explanation failed: {e})"
            st.write(f"**Why relevant:** {expl}")
        else:
            st.write(trim(abs_full))

        # ---------- Metadata dropdown per result ----------
        with st.expander("Show details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Abstract**")
                st.write(abs_full if isinstance(abs_full, str) and abs_full else "_(no abstract)_")
            with col2:
                pub = fmt_date(row.get("published", ""))
                upd = fmt_date(row.get("updated", ""))
                doi = row.get("doi", "") or row.get("doi_meta", "")
                arx = row.get(ARXIV_ID_COL, "") or row.get("arxiv_id", "")
                fn = row.get("filename", "") or row.get("filename_meta", "")
                st.markdown("**Metadata**")
                st.write(f"- **Published:** {pub or '—'}")
                st.write(f"- **Updated:** {upd or '—'}")
                st.write(f"- **DOI:** {doi or '—'}")
                st.write(f"- **arXiv ID:** {arx or '—'}")
                st.write(f"- **File:** {fn or Path(str(row.get(FILEPATH_COL,''))).name}")
                if arx:
                    st.write(f"- **arXiv:** https://arxiv.org/abs/{arx}")
                if doi:
                    st.write(f"- **Crossref:** https://doi.org/{doi}")

        st.divider()

    with st.expander("Raw table"):
        cols = [TITLE_COL, ABSTR_COL, "ann_cosine"] + (["rerank_cosine"] if "rerank_cosine" in cand.columns else [])
        # Include published/updated/doi if present
        for extra in ["published","updated","doi",ARXIV_ID_COL,FILEPATH_COL]:
            if extra in cand.columns and extra not in cols:
                cols.append(extra)
        st.dataframe(cand.head(showk)[cols].reset_index(drop=True))
else:
    st.info("Enter a query and press **Search** to see results.")
