from __future__ import annotations
import os
import pandas as pd
from typing import Optional

from app.spec_models import ExtractedFields, ExtractedDate

from .extractors import (
    extract_content,
    extract_authors,
    extract_title,
    extract_date_range,
)

# Cache metadata once
_metadata_cache: Optional[pd.DataFrame] = None


def _resolve_default_metadata_path() -> str:
    """
    Resolve the metadata file path relative to the project root,
    regardless of the Streamlit working directory.
    """
    # decompose.py lives in: ResearchAggregation/app/
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(APP_DIR)

    path = os.path.join(
        ROOT_DIR,
        "data2",
        "enhanced_final_meta_dataset_with_arxiv_deduped.csv"
    )
    return path


def load_metadata(path: str) -> pd.DataFrame:
    """
    Load metadata CSV with caching and normalization.
    Always receives an absolute path from the caller.
    """
    global _metadata_cache

    if _metadata_cache is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Metadata file not found at: {path}\n"
                "Ensure the path is correct or project structure is intact."
            )

        df = pd.read_csv(path)

        # Normalize fields expected by query logic
        df["paper_title_norm"] = df["paper_title"].astype(str).str.strip().str.lower()
        df["authors_norm"] = df["authors"].astype(str).str.lower()
        df["date_norm"] = df["date"].astype(str).str.lower()

        _metadata_cache = df

    return _metadata_cache


def search_metadata_for_matches(query: str, fields: ExtractedFields, meta: pd.DataFrame):
    """
    Apply simple metadata-based filtering based on title, authors, and date range.
    """
    # If content exists → it's a topic query → do NOT metadata-only search
    if fields.content.content:
        return None

    author_list = fields.authors.authors
    title = (fields.title.title or "").strip().lower()
    year_min = fields.date.year_min
    year_max = fields.date.year_max

    # If no metadata signals → fallback to semantic search
    if not author_list and not title and year_min is None and year_max is None:
        return None

    df = meta.copy()

    # -------------------------
    # 1. Title match
    # -------------------------
    if title:
        exact = df[df["paper_title_norm"] == title]
        if len(exact) > 0:
            return exact

        contains = df[df["paper_title_norm"].str.contains(title, na=False)]
        if len(contains) > 0:
            return contains

    # -------------------------
    # 2. Author match
    # -------------------------
    if author_list:
        tmp = df
        for a in [a.lower() for a in author_list]:
            tmp = tmp[tmp["authors_norm"].str.contains(a, na=False)]

        if len(tmp) > 0:
            df = tmp

    # -------------------------
    # 3. Date match
    # -------------------------
    if year_min is not None or year_max is not None:
        def extract_year(s: str):
            if not isinstance(s, str):
                return None
            import re
            m = re.search(r"(19|20)\d{2}", s)
            return int(m.group(0)) if m else None

        df["year_val"] = df["date"].apply(extract_year)

        if year_min is not None:
            df = df[df["year_val"].isna() | (df["year_val"] >= year_min)]

        if year_max is not None:
            df = df[df["year_val"].isna() | (df["year_val"] <= year_max)]

        if len(df) > 0:
            return df

    # Nothing matched strongly enough → fallback to semantic search
    return None


def decompose_query(
    query: str,
    metadata_path: str = None
) -> tuple[ExtractedFields, Optional[pd.DataFrame]]:
    """
    Extract metadata signals from a user query and optionally return
    a metadata-filtered subset of the dataset.

    If metadata_path is None, resolves a correct absolute path.
    """

    # Step 0 — choose correct metadata file
    if metadata_path is None:
        metadata_path = _resolve_default_metadata_path()
    else:
        # If user passes a relative path, resolve it relative to project root
        if not os.path.isabs(metadata_path):
            APP_DIR = os.path.dirname(os.path.abspath(__file__))
            ROOT_DIR = os.path.dirname(APP_DIR)
            metadata_path = os.path.join(ROOT_DIR, metadata_path)

    # Step 1 — extract all fields
    content = extract_content(query)
    authors = extract_authors(query)
    title = extract_title(query)
    date_range = extract_date_range(query)

    fields = ExtractedFields(
        content=content,
        authors=authors,
        title=title,
        date=date_range,
    )

    # Step 2 — load metadata file and attempt metadata lookup
    meta = load_metadata(metadata_path)
    matches = search_metadata_for_matches(query, fields, meta)

    return fields, matches
