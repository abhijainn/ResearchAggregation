import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import time

import streamlit as st
from langchain_community.vectorstores import FAISS

try:
    from app.config import CONFIG  # type: ignore
except ImportError:
    from config import CONFIG  # type: ignore

# Helper Modules
from helpers._get_abstract import *
from helpers._LLM import *
from helpers._cross_encoder_rerank import *
from helpers._load_data import *
from helpers._log import *


# -----------------------------
# CONFIG — edit paths if needed
# -----------------------------
TOPK_INITIAL = CONFIG.search.topk_initial
TOPK_SHOW = CONFIG.search.topk_show

DATA_DIR = CONFIG.paths.data_dir
INDEX_NAME = CONFIG.faiss.index_name
EMBED_MODEL_NAME = CONFIG.vector.embed_model_name


# -----------------------------
# Helpers
# -----------------------------

@lru_cache(maxsize=1)
def get_embedder(model_name: str = EMBED_MODEL_NAME) -> Specter2Embeddings:
    return Specter2Embeddings(model_name=model_name)


@lru_cache(maxsize=1)
def load_vectorstore() -> FAISS:
    index_path = os.path.join(DATA_DIR, f"{INDEX_NAME}.faiss")
    store_path = os.path.join(DATA_DIR, f"{INDEX_NAME}.pkl")
    if not (os.path.exists(index_path) and os.path.exists(store_path)):
        raise FileNotFoundError('Run the database creation step first.')

    embedder = get_embedder()
    log("Loaded vectorstore")
    return FAISS.load_local(
        DATA_DIR,
        embedder,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

def semantic_search(
    query: str,
    top_k: int = 5,
    fetch_k: int | None = None,
    *,
    use_hypothesis: bool = False,
    precomputed_hypothesis: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str]]:
    vectorstore = load_vectorstore()
    fetch = fetch_k or max(top_k, 20)
    fetch = max(fetch, top_k)

    search_text = query
    hypothesis: Optional[str] = None
    candidate_hypothesis: Optional[str] = precomputed_hypothesis
    if candidate_hypothesis is None and use_hypothesis:
        try:
            candidate_hypothesis = get_claim(query)
        except Exception as exc:
            raise RuntimeError(f"Failed to generate hypothesis: {exc}") from exc
    if isinstance(candidate_hypothesis, str):
        stripped = candidate_hypothesis.strip()
        if stripped:
            search_text = stripped
            hypothesis = stripped
        else:
            hypothesis = None
    else:
        hypothesis = None

    docs_with_scores = vectorstore.similarity_search_with_score(search_text, k=fetch)

    results = []
    for doc, score in docs_with_scores:
        doc_info = dict(doc.metadata)
        doc_info.setdefault('content', doc.page_content)
        doc_info['vector_score'] = float(score)
        results.append(doc_info)
    return results[:top_k], hypothesis

@st.cache_data(show_spinner=False)
def cached_summarize_title(title: str) -> Optional[str]:
    return summarize_title(title)


def _format_single_author(author: Any) -> Optional[str]:
    if not author:
        return None
    if isinstance(author, str):
        stripped = author.strip()
        return stripped or None
    if isinstance(author, dict):
        for key in ('name', 'full_name'):
            value = author.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        parts = [author.get('first'), author.get('middle'), author.get('last'), author.get('suffix')]
        joined = ' '.join(part for part in parts if isinstance(part, str) and part.strip())
        return joined or None
    return str(author)


def format_authors(authors: Any) -> Optional[str]:
    if not authors:
        return None
    if isinstance(authors, (list, tuple, set)):
        names = [_format_single_author(item) for item in authors]
        names = [name for name in names if name]
        return ', '.join(names) if names else None
    return _format_single_author(authors)


def format_date(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            formatted = format_date(item)
            if formatted:
                return formatted
        return None
    text = str(value).strip()
    if not text:
        return None
    if 'T' in text:
        text = text.split('T', 1)[0]
    return text


def pick_first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        else:
            text = str(value).strip()
            if text:
                return text
    return None


def render_result(rank: int, paper: Dict[str, Any]) -> None:
    title = pick_first_non_empty(
        paper.get('title'),
        paper.get('paper_title'),
        paper.get('name'),
        paper.get('content'),
    ) or f'Result {rank}'

    authors = format_authors(paper.get('authors') or paper.get('author') or paper.get('creators'))
    published = format_date(
        pick_first_non_empty(
            paper.get('published'),
            paper.get('updated'),
            paper.get('year'),
            paper.get('date'),
        )
    )
    arxiv_id = pick_first_non_empty(paper.get('arxiv_id'), paper.get('id'))
    url = pick_first_non_empty(paper.get('url'), paper.get('source_url'), paper.get('pdf_url'))
    vector_score = paper.get('vector_score')

    clean_title = " ".join(str(title).splitlines()).strip()
    st.markdown(f"#### {rank}. {clean_title}")

    meta_lines = []
    if authors:
        meta_lines.append(f"- **Authors:** {authors}")
    if published:
        meta_lines.append(f"- **Date:** {published}")
    if arxiv_id:
        meta_lines.append(f"- **arXiv ID:** {arxiv_id}")
    if url:
        meta_lines.append(f"- **Link:** [{url}]({url})")
    if isinstance(vector_score, (int, float)):
        meta_lines.append(f"- **Vector score:** {vector_score:.3f}")
    if meta_lines:
        st.markdown("\n".join(meta_lines))

    llm_summary = paper.get('llm_summary')
    if llm_summary:
        st.markdown("**LLM Summary**")
        st.write(llm_summary)

    abstract = pick_first_non_empty(paper.get('abstract'), paper.get('summary'), paper.get('content'))
    st.markdown("**Abstract**")
    if abstract:
        st.write(abstract)
    else:
        st.markdown("_No abstract available._")
    st.divider()




def main() -> None:
    st.set_page_config(page_title="Research Paper Recommender", layout="wide")
    st.title("Research Aggregation Search")
    st.caption("Run semantic retrieval over the local arXiv corpus with SPECTER2 embeddings.")

    with st.form("search_form"):
        query = st.text_area(
            "Query",
            value="",
            placeholder="e.g. Retrieval-augmented generation for biomedical question answering",
            height=120,
        )
        top_k_default = min(TOPK_SHOW, TOPK_INITIAL)
        top_k = st.slider(
            "Number of results (top_k)",
            min_value=1,
            max_value=TOPK_INITIAL,
            value=top_k_default if top_k_default >= 1 else 5,
            step=1,
        )
        col1, col2, col3 = st.columns(3, gap="small")
        with col1:
            use_hypothesis = st.checkbox(
                "Search with hypothesis",
                value=False,
                help="When selected, generate a hypothesis from your query and use it for retrieval.",
            )
        with col2:
            apply_rerank = st.checkbox(
                "Enable cross-encoder reranking",
                value=False,
                help="When selected, rerank retrieved papers using the cross-encoder.",
            )
        with col3:
            generate_summaries = st.checkbox(
                "Generate LLM summaries",
                value=False,
                help="When selected, summarize each result title with the LLM helper.",
            )
        submitted = st.form_submit_button("Search")

    results: List[Dict[str, Any]] = []
    hypothesis_text: Optional[str] = None
    hypothesis_elapsed: Optional[float] = None
    rerank_elapsed: Optional[float] = None
    llm_summary_elapsed: Optional[float] = None
    llm_summaries_generated = False
    query_text = query.strip()

    if submitted:
        if not query_text:
            st.warning("Please enter a query before searching.")
        else:
            fetch_k = max(top_k * 2, 20)
            hypothesis_generation_failed = False
            if use_hypothesis:
                start = time.perf_counter()
                with st.spinner("Generating abstract..."):
                    try:
                        hypothesis_text = get_claim(query_text)
                    except Exception as exc:
                        st.warning(f"Abstract generation failed: {exc}")
                        hypothesis_text = None
                        hypothesis_generation_failed = True
                    finally:
                        hypothesis_elapsed = time.perf_counter() - start

            try:
                with st.spinner("Searching the corpus..."):
                    results, hypothesis_text = semantic_search(
                        query_text,
                        top_k=top_k,
                        fetch_k=fetch_k,
                        use_hypothesis=use_hypothesis and not hypothesis_generation_failed,
                        precomputed_hypothesis=hypothesis_text,
                    )
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                return

            if apply_rerank and results:
                start = time.perf_counter()
                rerank_results: Optional[List[Dict[str, Any]]] = None
                with st.spinner("Reranking results..."):
                    try:
                        rerank_query = hypothesis_text if (use_hypothesis and hypothesis_text) else query_text
                        rerank_results = rerank_with_cross_encoder(rerank_query, results, top_k=top_k)
                    except Exception as exc:
                        st.warning(f"Cross-encoder rerank failed: {exc}")
                        results = results[:top_k]
                    finally:
                        rerank_elapsed = time.perf_counter() - start
                if rerank_results is None:
                    print("\n\nRerank failed\n\n")
                    st.info("Rerank failed.")
                else:
                    results = rerank_results

            if generate_summaries and results:
                start = time.perf_counter()
                llm_error: Optional[Exception] = None
                with st.spinner("Generating LLM summaries..."):
                    for paper in results:
                        title_for_summary = pick_first_non_empty(
                            paper.get('title'),
                            paper.get('paper_title'),
                            paper.get('name'),
                            paper.get('content'),
                        )
                        if not title_for_summary:
                            continue
                        try:
                            summary_text = cached_summarize_title(title_for_summary)
                        except Exception as exc:
                            llm_error = exc
                            break
                        if summary_text:
                            paper['llm_summary'] = summary_text
                            llm_summaries_generated = True
                llm_summary_elapsed = time.perf_counter() - start
                if llm_error is not None:
                    st.warning(f"LLM summarization stopped early: {llm_error}")

            if results:
                if hypothesis_elapsed is not None:
                    for paper in results:
                        paper['abstract_generation_seconds'] = hypothesis_elapsed
                if rerank_elapsed is not None:
                    for paper in results:
                        paper['rerank_seconds'] = rerank_elapsed

    if use_hypothesis and submitted:
        if hypothesis_elapsed is not None:
            st.metric("Abstract Generation Duration (s)", f"{hypothesis_elapsed:.3f}")
        if hypothesis_text:
            st.text_area(
                "Generated Hypothesis",
                value=hypothesis_text,
                height=120,
                key="generated_hypothesis_display",
            )
        else:
            st.info("No hypothesis was generated; the search used the original query.")

    if apply_rerank and submitted:
        if rerank_elapsed is not None:
            st.metric("Rerank Duration (s)", f"{rerank_elapsed:.3f}")
        else:
            st.info("Rerank step skipped because no results were available.")

    if generate_summaries and submitted:
        if llm_summary_elapsed is not None:
            st.metric("LLM Summary Duration (s)", f"{llm_summary_elapsed:.3f}")
            if not llm_summaries_generated:
                st.info("LLM summarization completed but did not return any summaries.")
        elif results:
            st.info("LLM summarization skipped because no valid titles were available.")
        else:
            st.info("LLM summarization skipped because no results were available.")

    if results:
        st.subheader(f"Top {len(results)} result{'s' if len(results) != 1 else ''}")
        for rank, paper in enumerate(results, start=1):
            render_result(rank, paper)

        st.divider()
    elif submitted and query_text:
        st.info("No results found. Try broadening the query or lowering the top_k value.")
    elif not submitted:
        st.info("Enter a query and click **Search** to retrieve relevant papers.")


if __name__ == "__main__":
    main()



# def _safe_norm(x: np.ndarray) -> np.ndarray:
#     x = x.astype("float32", copy=False)
#     norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
#     return np.ascontiguousarray(x / norms)

# @st.cache_resource
# def load_data_and_index(parquet_path: str, metadata_csv: str):
#     if not Path(parquet_path).exists():
#         st.error(f"Parquet not found: {parquet_path}")
#         st.stop()
#     df = pd.read_parquet(parquet_path)

#     # Ensure text columns
#     for col in (TITLE_COL, ABSTR_COL):
#         if col not in df.columns:
#             df[col] = ""
#     for col in (FILEPATH_COL, URL_COL, ARXIV_ID_COL):
#         if col not in df.columns:
#             df[col] = ""

#     # Optional metadata enrichment
#     meta = None
#     if Path(metadata_csv).exists():
#         meta = pd.read_csv(metadata_csv)
#         # Keep only relevant columns if present
#         keep = ["filepath","filename","arxiv_id","short_id","version","published","updated","doi","title","abstract"]
#         cols = [c for c in keep if c in meta.columns]
#         meta = meta[cols].copy()

#         # Prefer merge on filepath if available; else try arxiv_id
#         if FILEPATH_COL in df.columns and "filepath" in meta.columns:
#             df = df.merge(meta, how="left", left_on=FILEPATH_COL, right_on="filepath", suffixes=("", "_meta"))
#         elif ARXIV_ID_COL in df.columns and "arxiv_id" in meta.columns:
#             df = df.merge(meta, how="left", left_on=ARXIV_ID_COL, right_on="arxiv_id", suffixes=("", "_meta"))
#         # If title/abstract exist in both, keep primary df versions

#         # Fill missing text fields from metadata if parquet lacked them
#         if df[TITLE_COL].eq("").any() and "title_meta" in df.columns:
#             df[TITLE_COL] = df[TITLE_COL].mask(df[TITLE_COL].eq(""), df["title_meta"].fillna(""))
#         if df[ABSTR_COL].eq("").any() and "abstract_meta" in df.columns:
#             df[ABSTR_COL] = df[ABSTR_COL].mask(df[ABSTR_COL].eq(""), df["abstract_meta"].fillna(""))

#     # Embeddings
#     if EMBED_COL not in df.columns:
#         st.error(f"Parquet must contain an '{EMBED_COL}' column with arrays.")
#         st.stop()
#     X = np.vstack(df[EMBED_COL].to_numpy())
#     X = _safe_norm(X)
#     index = faiss.IndexFlatIP(X.shape[1])
#     index.add(X)

#     return df, index

# @st.cache_resource
# def load_specter2():
#     tok = AutoTokenizer.from_pretrained(SPECTER2_MODEL, use_fast=True)
#     model = AutoAdapterModel.from_pretrained(SPECTER2_MODEL)
#     model.load_adapter(SPECTER2_ADAPTER, source="hf", load_as="specter2")
#     model.set_active_adapters("specter2")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device).eval()
#     return tok, model, device

# @torch.inference_mode()
# def encode_query_specter2(tok, model, device, text: str) -> np.ndarray:
#     enc = tok(text, truncation=True, max_length=512, return_tensors="pt").to(device)
#     cls = model(**enc).last_hidden_state[:, 0]
#     cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
#     return cls[0].detach().cpu().numpy().astype("float32")[None, :]

# @st.cache_resource
# def load_reranker():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     return SentenceTransformer(RERANK_MODEL, device=device)

# def rerank_with_mxbai(query: str, doc_vectors: np.ndarray, rer_model, batch_size: int = 64):
#     qv = rer_model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
#     scores = doc_vectors @ qv.astype(np.float32)
#     return scores

# def trim(s: str, n=420):
#     s = (s or "").replace("\n", " ").strip()
#     return s if len(s) <= n else s[:n] + "…"

# def link_for_row(row):
#     url = row.get(URL_COL, "") or ""
#     if isinstance(url, str) and url:
#         return url
#     fp = row.get(FILEPATH_COL, "") or ""
#     if isinstance(fp, str) and fp:
#         return f"file://{Path(fp).absolute()}"
#     return ""

# def fmt_date(s):
#     if pd.isna(s) or not isinstance(s, str) or not s:
#         return ""
#     # s may already be ISO; keep as-is but shorten if needed
#     return s.split("T")[0] if "T" in s else s

# # ---- LLM explainer ----
# @st.cache_resource
# def load_llm():
#     use_gpu = torch.cuda.is_available()
#     name = LLM_GPU_MODEL if use_gpu else LLM_CPU_MODEL
#     tok = AutoTokenizer.from_pretrained(name, use_fast=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         name,
#         torch_dtype=torch.float16 if use_gpu else torch.float32,
#         device_map="auto" if use_gpu else None
#     )
#     model.eval()
#     return tok, model, name

# def build_prompt(query: str, title: str, abstract: str) -> str:
#     # Stronger, explicit instructions to the LLM to keep the reply short and to the point.
#     return (
#         "You are an expert in NLP research.\n"
#         "Task: In 2–3 concise sentences, explain WHY the given paper (title + abstract) is relevant to the user query.\n"
#         "Requirements: mention concrete overlaps (task, method, dataset, or key findings). Do NOT include background, filler, or speculative commentary. Use full sentences and be precise.\n\n"
#         f"Query: {query}\n\n"
#         f"Paper Title: {title}\n\n"
#         f"Paper Abstract: {abstract}\n\n"
#         "Answer (exactly 2–3 sentences):"
#     )

# @torch.inference_mode()
# def _first_n_sentences(text: str, n: int = 3) -> str:
#     """
#     Return the first `n` *complete* sentences from the text.
#     Truncates cleanly at sentence boundaries (., !, ?).
#     """
#     if not text:
#         return ""
#     text = text.strip().replace("\n", " ")
#     # Split at sentence enders with punctuation followed by whitespace
#     parts = re.split(r'(?<=[.!?])\s+', text)
#     parts = [s.strip() for s in parts if s.strip()]
#     selected = []
#     for part in parts:
#         selected.append(part)
#         if len(selected) >= n:
#             break
#     return " ".join(selected)

# def explain_relevance(tok, model, query: str, title: str, abstract: str, max_new_tokens: int = 80) -> str:
#     prompt = build_prompt(query, title or "", abstract or "")
#     inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
#     out_ids = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#         temperature=0.2,
#         repetition_penalty=1.05,
#         eos_token_id=tok.eos_token_id,
#     )
#     text = tok.decode(out_ids[0], skip_special_tokens=True)
#     if "Answer" in text:
#         text = text.split("Answer")[-1].lstrip(":").strip()
#     # Post-process: prefer 2 sentences, allow up to 3 if the second contains an abbreviation that ends with '.'
#     cleaned = text.strip()
#     # If the model repeated the prompt, remove prompt content occurrences (defensive)
#     if prompt.strip() in cleaned:
#         cleaned = cleaned.replace(prompt.strip(), "")
#     # Extract first 2 sentences, but if there are 2 very short sentences (<20 chars), try up to 3.
#     first2 = _first_n_sentences(cleaned, 3)
#     # If first2 is short, try 3 sentences to provide slightly more substance
#     if len(first2) < 20:
#         first3 = _first_n_sentences(cleaned, 4)
#         return first3.strip()
#     return first2.strip()

# -----------------------------
# UI (with session state patch)
# -----------------------------
# st.set_page_config(page_title="Local NLP Paper Search", layout="wide")
# st.title("🔎 Local NLP Paper Search")

# with st.sidebar:
#     st.header("Settings")
#     topk = st.slider("Initial ANN top‑k", 10, 200, TOPK_INITIAL, 10)
#     showk = st.slider("Show top‑k", 5, 50, TOPK_SHOW, 5)
#     do_rerank = st.checkbox("Re‑rank with mxbai embeddings", value=True)
#     gen_llm = st.checkbox("Generate ‘why relevant’ (open‑source LLM)", value=True)
#     st.caption("Disable re‑ranking and LLM for faster results.")

# query = st.text_input("Query (e.g., “Large language models for automatic speech recognition”):", value="")
# search = st.button("Search", type="primary", use_container_width=True)

# # Load data and models
# df, index = load_data_and_index(SPECTER2_PARQUET, METADATA_CSV)
# tok_s2, model_s2, device_s2 = load_specter2()
# rer_model = load_reranker() if do_rerank else None
# tok_llm = model_llm = model_llm_name = None
# if gen_llm:
#     tok_llm, model_llm, model_llm_name = load_llm()

# # Load cached doc vectors
# MXBAI_VECS_PATH = "app/mxbai_doc_vectors.npy"
# mxbai_vecs = np.load(MXBAI_VECS_PATH, mmap_mode="r").astype(np.float32)

# # Handle query + search
# if search and query.strip():
#     with st.spinner("Encoding query with SPECTER2 and searching ANN…"):
#         q = encode_query_specter2(tok_s2, model_s2, device_s2, query.strip())
#         D, I = index.search(q, topk)

#     cand = df.iloc[I[0]].copy()
#     cand["ann_cosine"] = D[0]

#     if do_rerank:
#         with st.spinner("Re‑ranking with mxbai embeddings…"):
#             doc_subvecs = mxbai_vecs[I[0]]
#             r_scores = rerank_with_mxbai(query.strip(), doc_subvecs, rer_model)
#             cand["rerank_cosine"] = r_scores
#             cand = cand.sort_values("rerank_cosine", ascending=False)
#     else:
#         cand = cand.sort_values("ann_cosine", ascending=False)

#     # Save to session state
#     st.session_state["last_query"] = query.strip()
#     st.session_state["last_results"] = cand.reset_index(drop=True)

# # Use saved results if available
# cur_query = query.strip() or st.session_state.get("last_query", "")
# cur_results = st.session_state.get("last_results", None)

# if cur_query and cur_results is not None:
#     st.subheader("Results")
#     for i, row in cur_results.head(showk).iterrows():
#         title = row[TITLE_COL] or "(untitled)"
#         abs_full = row[ABSTR_COL]
#         ann_s = f"{row['ann_cosine']:.3f}"

#         st.markdown(f"### {i+1}. {title}")
#         if "rerank_cosine" in row:
#             st.caption(f"ANN cosine: {ann_s}  |  Re‑rank cosine: {row['rerank_cosine']:.3f}")
#         else:
#             st.caption(f"ANN cosine: {ann_s}")

#         if gen_llm and tok_llm is not None:
#             with st.expander("Why relevant (LLM)", expanded=False):
#                 key = f"llm_button_{i}"
#                 if st.button("Generate explanation", key=key):
#                     with st.spinner("Generating explanation..."):
#                         try:
#                             expl = explain_relevance(tok_llm, model_llm, cur_query, title, abs_full)
#                         except Exception as e:
#                             expl = f"(LLM explanation failed: {e})"
#                         st.session_state[f"llm_output_{i}"] = expl

#                 if f"llm_output_{i}" in st.session_state:
#                     st.write(st.session_state[f"llm_output_{i}"])


#         with st.expander("Show details"):
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("**Abstract**")
#                 st.write(abs_full if isinstance(abs_full, str) and abs_full else "_(no abstract)_")
#             with col2:
#                 pub = fmt_date(row.get("published", ""))
#                 upd = fmt_date(row.get("updated", ""))
#                 doi = row.get("doi", "") or row.get("doi_meta", "")
#                 arx = row.get(ARXIV_ID_COL, "") or row.get("arxiv_id", "")
#                 fn = row.get("filename", "") or row.get("filename_meta", "")
#                 st.markdown("**Metadata**")
#                 st.write(f"- **Published:** {pub or '—'}")
#                 st.write(f"- **Updated:** {upd or '—'}")
#                 st.write(f"- **DOI:** {doi or '—'}")
#                 st.write(f"- **arXiv ID:** {arx or '—'}")
#                 st.write(f"- **File:** {fn or Path(str(row.get(FILEPATH_COL,''))).name}")
#                 if arx:
#                     st.write(f"- **arXiv:** https://arxiv.org/abs/{arx}")
#                 if doi:
#                     st.write(f"- **Crossref:** https://doi.org/{doi}")

#         st.divider()

#     with st.expander("Raw table"):
#         cols = [TITLE_COL, ABSTR_COL, "ann_cosine"] + (["rerank_cosine"] if "rerank_cosine" in cur_results.columns else [])
#         for extra in ["published", "updated", "doi", ARXIV_ID_COL, FILEPATH_COL]:
#             if extra in cur_results.columns and extra not in cols:
#                 cols.append(extra)
#         st.dataframe(cur_results.head(showk)[cols].reset_index(drop=True))
# else:
#     st.info("Enter a query and press **Search** to see results.")
