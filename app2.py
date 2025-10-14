import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import torch
import torch.nn.functional as F
from adapters import AutoAdapterModel
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer

# Helper Modules
from modify_prompt import *
from cross_encoder_rerank import rerank_with_cross_encoder

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None



# -----------------------------
# CONFIG — edit paths if needed
# -----------------------------
SPECTER2_PARQUET = "app/arxiv_specter2_embeddings.parquet"  # must have 'embedding','title','abstract'; optional 'filepath','url','arxiv_id'
MXBAI_VECS_PATH = "app/mxbai_doc_vectors.npy"
METADATA_CSV = "app/arxiv_metadata.csv"                     # optional; adds published/updated/doi/filename
TITLE_COL = "title"
ABSTR_COL = "abstract"
EMBED_COL = "embedding"
FILEPATH_COL = "filepath"
URL_COL = "url"
ARXIV_ID_COL = "arxiv_id"

TOPK_INITIAL = 50
TOPK_SHOW = 15

DATA_DIR = 'data'
JSONL_PATH = os.path.join(DATA_DIR, 'arxiv_cs_only.jsonl')
INDEX_NAME = 'faiss'
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.json')
EMBED_MODEL_NAME = 'allenai/specter2_base'
RERANK_MODEL_NAME = 'mixedbread-ai/mxbai-rerank-large-v2'



# Models
SPECTER2_MODEL = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"
RERANK_MODEL = "mixedbread-ai/mxbai-embed-large-v1"  # embedding reranker via cosine
LLM_GPU_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_CPU_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# -----------------------------
# Helpers
# -----------------------------

class Specter2Embeddings:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.model.load_adapter('allenai/specter2', source='hf', load_as='proximity', set_active=True)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_inputs(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False,
            max_length=512,
        )

    def _embed_batch(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        normalized = F.normalize(cls_embeddings, p=2, dim=1)
        return normalized

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[List[float]] = []
        total = len(texts)
        use_progress_bar = tqdm is not None and total > self.batch_size
        progress_bar = tqdm(total=total, desc='Embedding texts', unit='doc') if use_progress_bar else None
        log_progress = tqdm is None and total > self.batch_size
        log_step = max(self.batch_size, total // 10 or 1) if log_progress else None
        next_log = log_step if log_progress else None
        if log_progress:
            print(f'Embedding {total} texts...')
        processed = 0
        try:
            with torch.inference_mode():
                for start_idx in range(0, total, self.batch_size):
                    batch = texts[start_idx:start_idx + self.batch_size]
                    inputs = self._prepare_inputs(batch)
                    normalized = self._embed_batch(inputs)
                    embeddings.extend(normalized.cpu().tolist())
                    batch_size = len(batch)
                    processed += batch_size
                    if progress_bar is not None:
                        progress_bar.update(batch_size)
                    elif log_progress and processed >= next_log:
                        print(f'Embedded {processed}/{total} texts')
                        next_log += log_step
        finally:
            if progress_bar is not None:
                progress_bar.close()
        if log_progress:
            if processed < total:
                print(f'Embedded {processed}/{total} texts')
            print('Embedding complete.')
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]


    def __call__(self, texts):
        if isinstance(texts, str):
            return self.embed_query(texts)
        try:
            iterable = list(texts)
        except TypeError as exc:
            raise TypeError('Expected iterable of strings or a single string.') from exc
        return self.embed_documents(iterable)

@lru_cache(maxsize=1)
def get_embedder(model_name: str = EMBED_MODEL_NAME) -> Specter2Embeddings:
    return Specter2Embeddings(model_name=model_name)


def prepare_corpus_texts(papers: List[Dict], sep_token: str) -> List[str]:
    prepared: List[str] = []
    for paper in papers:
        title = (paper.get('title') or '').strip()
        abstract = (paper.get('abstract') or '').strip()
        joined = f"{title}{sep_token}{abstract}" if title or abstract else ''
        prepared.append(joined)
    return prepared

def load_vectorstore() -> FAISS:
    index_path = os.path.join(DATA_DIR, f"{INDEX_NAME}.faiss")
    store_path = os.path.join(DATA_DIR, f"{INDEX_NAME}.pkl")
    if not (os.path.exists(index_path) and os.path.exists(store_path)):
        raise FileNotFoundError('Run the database creation step first.')

    embedder = get_embedder()
    return FAISS.load_local(
        DATA_DIR,
        embedder,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

@lru_cache(maxsize=1)
def get_vectorstore_cached() -> FAISS:
    return load_vectorstore()

def semantic_search(
    query: str,
    top_k: int = 5,
    fetch_k: int | None = None,
    *,
    use_hypothesis: bool = False,
) -> Tuple[List[Dict], Optional[str]]:
    vectorstore = get_vectorstore_cached()
    fetch = fetch_k or max(top_k, 20)
    fetch = max(fetch, top_k)

    search_text = query
    hypothesis: Optional[str] = None
    if use_hypothesis:
        try:
            hypothesis = get_claim(query)
        except Exception as exc:
            raise RuntimeError(f"Failed to generate hypothesis: {exc}") from exc
        if isinstance(hypothesis, str):
            stripped = hypothesis.strip()
            search_text = stripped or query
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
    cross_score = paper.get('cross_score')

    st.markdown(f"<h4 style='white-space:normal; line-height:1.3; font-size:1.2rem'>{title}</h4>", unsafe_allow_html=True)

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
    if isinstance(cross_score, (int, float)):
        meta_lines.append(f"- **Cross-encoder score:** {cross_score:.4f}")
    if meta_lines:
        st.markdown("\n".join(meta_lines))

    abstract = pick_first_non_empty(paper.get('abstract'), paper.get('summary'), paper.get('content'))
    st.markdown("**Abstract**")
    if abstract:
        st.write(abstract)
    else:
        st.markdown("_No abstract available._")
    st.divider()


def main() -> None:
    st.set_page_config(page_title="Research Aggregation Search", layout="wide")
    st.title("Research Aggregation Search")
    st.caption("Run semantic retrieval over the local arXiv corpus with SPECTER2 embeddings.")

    with st.form("search_form"):
        query = st.text_area(
            "Query",
            value="",
            placeholder="e.g. Retrieval-augmented generation for biomedical question answering",
            height=120,
        )

        use_hypothesis = st.checkbox(
            "Search with hypothesis",
            value=False,
            help="When selected, generate a hypothesis from your query and use it for retrieval.",
        )

        use_rerank = st.checkbox(
            "Enable cross-encoder reranking",
            value=False,
            help="When enabled, rerank a larger set of retrieved candidates using a cross-encoder (slower but more accurate).",
        )

        if not use_rerank:
            top_k_default = min(TOPK_SHOW, TOPK_INITIAL)
            top_k = st.slider(
                "Number of results (top_k)",
                min_value=1,
                max_value=TOPK_INITIAL,
                value=top_k_default if top_k_default >= 1 else 5,
                step=1,
            )

        if use_rerank:
            index_fetch_k = st.number_input(
                "Number of candidates to fetch for reranking (index_top_k)",
                min_value=1,
                max_value=TOPK_INITIAL,
                value=50,
                step=1,
            )
            display_k = st.slider(
                "Number of results to display after reranking",
                min_value=1,
                max_value=TOPK_SHOW,
                value=min(15, TOPK_SHOW),
                step=1,
            )

        submitted = st.form_submit_button("Search")


    results: List[Dict[str, Any]] = []
    hypothesis_text: Optional[str] = None
    query_text = query.strip()

    if submitted:
        if not query_text:
            st.warning("Please enter a query before searching.")
        else:
            # Non-rerank flow: fetch and display `top_k` results
            if not use_rerank:
                fetch_k = max(top_k * 2, 20)
                with st.spinner("Searching the corpus..."):
                    try:
                        results, hypothesis_text = semantic_search(
                            query_text,
                            top_k=top_k,
                            fetch_k=fetch_k,
                            use_hypothesis=use_hypothesis,
                        )
                    except FileNotFoundError as exc:
                        st.error(str(exc))
                        return
                    except Exception as exc:
                        st.error(f"Search failed: {exc}")
                        return

    # If reranking is enabled, fetch `index_fetch_k` results from the index and rerank to `display_k`
    if use_rerank and submitted and query_text:
        try:
            index_fetch_k = int(index_fetch_k)
        except Exception:
            index_fetch_k = 50
        try:
            display_k = int(display_k)
        except Exception:
            display_k = min(15, TOPK_SHOW)

        with st.spinner("Fetching candidates and reranking with cross-encoder..."):
            try:
                results, hypothesis_text = semantic_search(
                    query_text,
                    top_k=index_fetch_k,
                    fetch_k=index_fetch_k,
                    use_hypothesis=use_hypothesis,
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                return

            if results:
                try:
                    results = rerank_with_cross_encoder(query_text, results, top_k=display_k)
                except Exception as exc:
                    st.warning(f"Cross-encoder rerank failed: {exc}")
                    results = results[:display_k]

    if use_hypothesis and submitted:
        if hypothesis_text:
            st.text_area(
                "Generated Hypothesis",
                value=hypothesis_text,
                height=120,
                key="generated_hypothesis_display",
            )
        else:
            st.info("No hypothesis was generated; the search used the original query.")

    if results:
        st.subheader(f"Top {len(results)} result{'s' if len(results) != 1 else ''}")
        for rank, paper in enumerate(results, start=1):
            render_result(rank, paper)
    elif submitted and query_text:
        st.info("No results found. Try broadening the query or lowering the top_k value.")
    elif not submitted:
        st.info("Enter a query and click **Search** to retrieve relevant papers.")


if __name__ == "__main__":
    main()