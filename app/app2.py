import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import time
from QuerySpec import analyze_query_llm, simple_post_filter, QuerySpec

import streamlit as st
import numpy as np
import pandas as pd
from numpy.linalg import norm

import pyarrow.parquet as pq

from helpers import *

try:
    from app.config import CONFIG  # type: ignore
except ImportError:
    from app.config import CONFIG  # type: ignore

# Helper Modules
from helpers._get_abstract import *
from helpers._LLM import *
from helpers._cross_encoder_rerank import *
from helpers._load_data import *
from helpers._log import *

from app.decompose import decompose_query

import pandas as pd

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
def load_embedding_corpus():
    """
    Loads embeddings + IDs from the merged Parquet file instead of .npy/.csv.

    Uses:
      - specter2_embeddings_all.parquet  (merged batch files)
      - enhanced_final_meta_dataset_with_arxiv_deduped.csv

    Returns:
      embeddings: np.ndarray of shape [N, D]
      merged: pd.DataFrame with enriched metadata aligned to embedding rows
    """

    parquet_path = r"C:\Users\sophi\Documents\research\FDL research\ResearchAggregation\data2\specter2_embeddings_all.parquet"
    enhanced_path = r"C:\Users\sophi\Documents\research\FDL research\ResearchAggregation\data2\enhanced_final_meta_dataset_with_arxiv_deduped.csv"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing merged parquet file: {parquet_path}")
    if not os.path.exists(enhanced_path):
        raise FileNotFoundError(f"Missing enhanced metadata CSV: {enhanced_path}")

    # ====================================================
    # 1. Load Parquet (embeddings + IDs)
    # ====================================================
    table = pq.read_table(parquet_path)

    # Convert to pandas for easier merging logic
    df_emb = table.to_pandas()

    # Normalize ID column to string
    if "source_id_clean" not in df_emb.columns:
        raise KeyError("Column 'source_id_clean' missing from parquet file")

    df_emb["source_id_clean"] = df_emb["source_id_clean"].astype(str).str.strip()

    # Extract embeddings as a numpy array
    # Each row is a list<float>, so convert manually
    embeddings = np.vstack(df_emb["embedding"].apply(lambda x: np.array(x, dtype=np.float32)))

    # ====================================================
    # 2. Load enhanced metadata
    # ====================================================
    enhanced = pd.read_csv(enhanced_path)

    # Normalize ID formats
    enhanced["source_id_clean"] = enhanced["source_id_clean"].astype(str).str.strip()

    # Debug intersections
    set_emb = set(df_emb["source_id_clean"])
    set_enh = set(enhanced["source_id_clean"])

    print("Embedding IDs:", len(set_emb))
    print("Enhanced IDs:", len(set_enh))
    print("Intersection:", len(set_emb & set_enh))
    print("Example overlap:", list(set_emb & set_enh)[:10])

    missing_in_enh = set_emb - set_enh
    missing_in_emb = set_enh - set_emb
    print("Missing in enhanced:", len(missing_in_enh))
    print("Missing in embeddings:", len(missing_in_emb))

    # ====================================================
    # 3. Merge by source_id_clean
    # Keeps embedding order because we merge df_emb → enhanced
    # ====================================================
    merged = df_emb.merge(
        enhanced,
        on="source_id_clean",
        how="left",
        suffixes=("", "_enh"),
    )

    # ====================================================
    # 4. Fill missing metadata fields
    # ====================================================
    def _fill_missing(target_col, source_col):
        if target_col not in merged.columns or source_col not in merged.columns:
            return
        mask = merged[target_col].isna() | merged[target_col].astype(str).str.strip().eq("")
        merged.loc[mask, target_col] = merged.loc[mask, source_col]

    _fill_missing("paper_title", "paper_title_enh")
    _fill_missing("authors", "authors_enh")

    # Other enhanced-only fields are already present under their names.

    return embeddings, merged

def build_results_from_metadata(df: pd.DataFrame) -> list[dict]:
    results = []
    for i, row in df.iterrows():
        entry = {
            "paper_title": row.get("paper_title"),
            "authors": row.get("authors"),
            "abstract": row.get("abstract"),
            "full_text": row.get("full_text"),
            "categories": row.get("categories"),
            "doi": row.get("doi"),
            "journal": row.get("journal"),
            "date": row.get("date"),
            "content": row.get("text"),
            "source_id": row.get("source_id_clean"),
            "vector_score": None,
            "row_index": row.get("row_index"),
        }
        results.append(entry)
    return results


def semantic_search(
    query: str,
    top_k: int = 5,
    fetch_k: int | None = None,
    *,
    use_hypothesis: bool = False,
    precomputed_hypothesis: Optional[str] = None,
    restrict_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Semantic search using:
      - specter2_embeddings_acl.npy  (NumPy array, shape [N, D])
      - specter2_id_map_acl.csv      (row_index, source_id, paper_title, authors, text)
      - enriched metadata from enhanced_final_meta_dataset_with_arxiv_deduped.csv
    """

    # Load embedding matrix + pandas mapping (enriched)
    embeddings, mapping = load_embedding_corpus()

    if restrict_df is not None:
        # Keep only rows whose source_id_clean appear in the restricted set
        restrict_ids = set(str(x).strip() for x in restrict_df["source_id"])
        mask = mapping["source_id"].astype(str).str.strip().isin(restrict_ids)

        # Filter embeddings and mapping together
        embeddings = embeddings[mask.values]
        mapping = mapping[mask.values]

    embedder = get_embedder()

    fetch = fetch_k or max(top_k, 20)
    fetch = max(fetch, top_k)

    # ----------------------------------------
    # Hypothesis generation handling
    # ----------------------------------------
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

    # ----------------------------------------
    # Encode the query using SPECTER2
    # ----------------------------------------
    q = embedder.embed_query(search_text)       # shape [D]
    q = q / (np.linalg.norm(q) + 1e-12)

    # Normalize embeddings for cosine similarity
    E = embeddings
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    # Compute cosine similarity
    sims = E @ q  # shape [N]

    # Select top fetch candidates
    idx = np.argpartition(sims, -fetch)[-fetch:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    idx = idx[:top_k]

    # ----------------------------------------
    # Build results (enriched) in the SAME general format as before
    # ----------------------------------------
    results: List[Dict[str, Any]] = []
    for i in idx:
        row = mapping.iloc[i]

        base_entry: Dict[str, Any] = {
            "paper_title": row.get("paper_title"),
            "authors": row.get("authors"),
            "date": row.get("date"),
            "content": row.get("text") or row.get("abstract"),
            "source_id": row.get("source_id"),
            "vector_score": float(sims[i]) if sims is not None else None,
        }

        results.append(base_entry)

    return results, hypothesis


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

    # -----------------------------------------
    # NEW: semicolon-delimited authors
    # -----------------------------------------
    if isinstance(authors, str) and ";" in authors:
        split_authors = [a.strip() for a in authors.split(";") if a.strip()]
        if not split_authors:
            return None
        return ", ".join(split_authors)

    # -----------------------------------------
    # EXISTING logic for list / tuple / set
    # -----------------------------------------
    if isinstance(authors, (list, tuple, set)):
        names = [_format_single_author(item) for item in authors]
        names = [name for name in names if name]
        return ', '.join(names) if names else None

    # Fallback for other str or object types
    return _format_single_author(authors)

def truncate_authors_list(author_string: str, limit: int = 5) -> Tuple[str, Optional[str]]:
    """
    Splits the author string into a list, truncates to `limit` authors,
    and returns (display_string, full_string_if_truncated).

    If not truncated, second return value is None.
    """
    authors = [a.strip() for a in author_string.split(",") if a.strip()]
    if len(authors) <= limit:
        return ", ".join(authors), None

    truncated = ", ".join(authors[:limit])
    full = ", ".join(authors)
    return truncated + " ...", full


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
        truncated, full_list = truncate_authors_list(authors, limit=5)

        if full_list is None:
            # No truncation needed — display normally
            meta_lines.append(f"- **Authors:** {truncated}")
        else:
            # Truncated — show short version, plus a Streamlit expander
            meta_lines.append(f"- **Authors:** {truncated}")

            with st.expander("Show full author list"):
                st.write(full_list)

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


def metadata_filter_enhanced(results: List[Dict[str, Any]], spec: QuerySpec) -> List[Dict[str, Any]]:
    """
    Apply metadata filters with special handling for date and journal:

      - If a date range is specified (via year_min/year_max), drop papers whose known
        dates fall outside the range. If a paper has no valid date, keep it (treat as
        unknown / "all the same" when no valid dates).
      - For journal/venue filters, require a match only when journal is known.
        Papers with missing journal are kept.

    Other filters (if any) can still be applied by simple_post_filter before/after,
    """
    filtered = list(results)

    # Extract potential numeric year filters from spec.metadata_filters
    year_min = None
    year_max = None
    requested_journal = None

    if getattr(spec, "metadata_filters", None):
        mf = spec.metadata_filters
        year_min = mf.get("year_min")
        year_max = mf.get("year_max")
        # 'venue' is a common key used for journal-like filtering
        requested_journal = mf.get("venue") or mf.get("journal")

    # --------------------------
    # Date Filtering (by year)
    # --------------------------
    if year_min is not None or year_max is not None:
        tmp: List[Dict[str, Any]] = []
        for r in filtered:
            date_val = r.get("date")

            # If we have no date, keep the paper (unknown date treated as "all the same")
            if date_val is None or (isinstance(date_val, float) and pd.isna(date_val)):
                tmp.append(r)
                continue

            # date can be "YYYY", "YYYY-MM", "YYYY-MM-DD"
            try:
                year_str = str(date_val).split("-")[0]
                year = int(year_str)
            except Exception:
                # Unparseable date → treat as unknown, keep it
                tmp.append(r)
                continue

            if year_min is not None and year < year_min:
                continue
            if year_max is not None and year > year_max:
                continue
            tmp.append(r)

        filtered = tmp

    # --------------------------
    # Journal / Venue Filtering
    # --------------------------
    if requested_journal:
        tmp = []
        req = str(requested_journal).lower().strip()
        for r in filtered:
            journal = r.get("journal")
            if journal is None or (isinstance(journal, float) and pd.isna(journal)):
                # Missing journal → keep (do not penalize unknowns)
                tmp.append(r)
                continue

            journal_str = str(journal).lower()
            if req in journal_str:
                tmp.append(r)
                continue

            # If no match, drop
        filtered = tmp

    return filtered


def main() -> None:
    st.set_page_config(page_title="Research Paper Recommender", layout="wide")

    st.title("Research Aggregation Search")
    st.caption("Run semantic retrieval over the local arXiv corpus with specter embeddings.")

    # -------------------------------
    # Two-column layout
    # -------------------------------
    left_col, right_col = st.columns([2, 3])

    # -------------------------------
    # CHAT WINDOW IN LEFT COLUMN
    # -------------------------------
    with left_col:
        
        st.markdown(
            """
            <style>
            .chat-box {
                height: 45vh;              /* fixed height */
                overflow-y: auto;          /* scroll inside */
                padding-right: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Chat scroll container
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        chat_container = st.container()    # <-- this is the same container you use later
        st.markdown("</div>", unsafe_allow_html=True)

        # Input box (static)
        with st.form("chat_input_form", clear_on_submit=True):
            user_query = st.text_area(
                "",
                height=90,
                placeholder="Ask a research question...",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Send")

    # -------------------------------
    # SIDEBAR SETTINGS
    # -------------------------------
    with st.sidebar:
        st.header("Search Options")

        top_k_default = min(TOPK_SHOW, TOPK_INITIAL)
        top_k = st.slider(
            "Number of results (top_k)",
            min_value=1,
            max_value=TOPK_INITIAL,
            value=top_k_default if top_k_default >= 1 else 5,
            step=1,
        )
        use_hypothesis = st.checkbox("Search with hypothesis", value=False)
        apply_rerank = st.checkbox("Enable cross-encoder reranking", value=False)
        generate_summaries = st.checkbox("Generate LLM summaries", value=False)
        use_llm_analyzer = st.checkbox("Use LLM query analyzer", value=True)

    # -------------------------------
    # STATE
    # -------------------------------
    results = []
    hypothesis_text = None
    hypothesis_elapsed = None
    rerank_elapsed = None
    llm_summary_elapsed = None
    llm_summaries_generated = False

    query_text = user_query.strip()

    # -------------------------------
    # HANDLE SEND
    # -------------------------------
    if send and query_text:

        # ----------------------------------------
        # (NEW) Metadata-Aware Decomposition
        # ----------------------------------------
        fields, metadata_matches = decompose_query(query_text)

        only_metadata = (
            metadata_matches is not None
            and len(metadata_matches) > 0
            and not fields.content.content  # empty content = pure metadata query
        )

        if only_metadata:
            # Fast-path: Return metadata matches (NO semantic search)
            results = build_results_from_metadata(metadata_matches)

            with left_col:
                with chat_container:
                    st.markdown(
                        f"""
                        <div style='background:#1e1e1e;padding:10px;border-radius:8px;margin-bottom:10px;'>
                            <b>System:</b> Found {len(results)} metadata match(es) — skipping semantic search.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with right_col:
                st.subheader("Search Results (Metadata Match)")
                for rank, paper in enumerate(results, start=1):
                    render_result(rank, paper)

            return  # STOP HERE


        with left_col:
            with chat_container:
                st.markdown(
                    f"""
                    <div style='background:#303030;padding:10px;border-radius:8px;margin-bottom:10px;'>
                        <b>You:</b> {query_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------
        # ANALYZER, SEARCH, RERANK, SUMMARIES
        # -------------------------------
        fetch_k = max(top_k * 2, 20)
        hypothesis_generation_failed = False
        spec = None
        semantic_text = query_text
        keyword_text = query_text

        # Analyzer
        try:
            spec = analyze_query_llm(query_text)
            if spec and spec.semantic_query:
                semantic_text = spec.semantic_query
            if spec and spec.keyword_query:
                keyword_text = spec.keyword_query
        except:
            pass

        # Hypothesis
        if use_hypothesis:
            start = time.perf_counter()
            with st.spinner("Generating hypothesis..."):
                try:
                    hypothesis_text = get_claim(query_text)
                except Exception:
                    hypothesis_text = None
                    hypothesis_generation_failed = True
                finally:
                    hypothesis_elapsed = time.perf_counter() - start

        # Search
        try:
            with st.spinner("Searching the corpus..."):
                results, hypothesis_text = semantic_search(
                    semantic_text,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    use_hypothesis=use_hypothesis and not hypothesis_generation_failed,
                    precomputed_hypothesis=hypothesis_text,
                )
                if spec and results:
                    results = metadata_filter_enhanced(results, spec)
        except Exception as exc:
            with left_col:
                with chat_container:
                    st.error(f"Search failed: {exc}")
            return

        # Reranking
        if apply_rerank and results:
            start = time.perf_counter()
            with st.spinner("Reranking results..."):
                try:
                    rerank_query = hypothesis_text if (use_hypothesis and hypothesis_text) else semantic_text
                    results = rerank_with_cross_encoder(rerank_query, results, top_k=top_k)
                except:
                    st.warning("Cross-encoder rerank failed.")
                finally:
                    rerank_elapsed = time.perf_counter() - start

        # LLM summaries
        if generate_summaries and results:
            start = time.perf_counter()
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
                    except Exception:
                        break
                    if summary_text:
                        paper["llm_summary"] = summary_text
                        llm_summaries_generated = True
            llm_summary_elapsed = time.perf_counter() - start

        # -------------------------------
        # LEFT COLUMN: SYSTEM MESSAGE
        # -------------------------------
        with left_col:
            with chat_container:
                st.markdown(
                    f"""
                    <div style='background:#1e1e1e;padding:10px;border-radius:8px;margin-bottom:10px;'>
                        <b>System:</b> Found {len(results)} result(s)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------
    # RIGHT COLUMN: ONLY SHOW IF RESULTS EXIST
    # -------------------------------
    if results:
        with right_col:
            st.subheader("Search Results")
            for rank, paper in enumerate(results, start=1):
                render_result(rank, paper)

    elif send and not query_text:
        with left_col:
            st.warning("Please enter a query before searching.")


if __name__ == "__main__":
    main()
