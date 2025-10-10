#!/usr/bin/env python3
"""
Read the first N entries from anthology+abstracts.bib, resolve each ACL Anthology URL
to its paper, fetch the PDF, extract full text, and save metadata + text to JSONL.

Requires:
  pip install acl-anthology requests beautifulsoup4 pymupdf

Note: The first call to Anthology.from_repo() downloads ~120MB of metadata locally
      and then reuses it on subsequent runs. (See docs.)
"""

import argparse
import json
import os
import re
import time
from typing import Iterator, Dict, Optional
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from acl_anthology import Anthology  # official library

# ---------- Streaming .bib: yield bibkey + URL without loading the whole file ----------
def iter_bib_urls(path: str, limit: Optional[int] = None) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        buf, depth, in_entry = [], 0, False
        remaining = limit
        for line in f:
            if not in_entry and line.lstrip().startswith('@'):
                in_entry, buf = True, [line]
                depth = line.count('{') - line.count('}')
                continue
            if in_entry:
                buf.append(line)
                depth += line.count('{') - line.count('}')
                if depth <= 0:
                    entry = ''.join(buf)
                    m_key = re.search(r'@\w+\s*\{\s*([^,]+)\s*,', entry)
                    m_url = re.search(r'\burl\s*=\s*(?:\{([^}]*)\}|"([^"]*)")', entry, re.I)
                    if m_key and m_url:
                        bibkey = m_key.group(1).strip()
                        url = (m_url.group(1) or m_url.group(2)).strip()
                        yield {"bibkey": bibkey, "url": url}
                        if remaining is not None:
                            remaining -= 1
                            if remaining <= 0:
                                return
                    in_entry, buf, depth = False, [], 0

def anthology_id_from_url(page_url: str) -> Optional[str]:
    # e.g. https://aclanthology.org/K19-1011/  -> "K19-1011"
    path = urlparse(page_url).path.strip("/")
    return path.split("/")[0] if path else None

# ---------- HTTP session with polite retries/backoff ----------
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ACL-Downloader/1.0)"})
    retry = Retry(
        total=5,
        backoff_factor=1.0,           # 1s, 2s, 4s, ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def download_pdf(url: str, out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, url.rstrip("/").split("/")[-1])
    with make_session().get(url, timeout=30, stream=True) as r:
        if r.status_code != 200 or "application/pdf" not in r.headers.get("Content-Type", ""):
            return None
        with open(fn, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
    return fn

def extract_text_pymupdf(pdf_path: str) -> str:
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # "text" is fine here; use "blocks" or "rawdict" if you want extra structure
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="anthology+abstracts.bib")
    ap.add_argument("--out", default="acl_text.jsonl")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--pdf_dir", default="pdfs")
    ap.add_argument("--keep_pdf", action="store_true", help="Keep downloaded PDFs (default: delete after extraction)")
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between PDF downloads")
    args = ap.parse_args()

    # 1) Instantiate Anthology (first time downloads ~120MB metadata; then local)
    anth = Anthology.from_repo()  # metadata is local after first run
    # Docs: https://acl-anthology.readthedocs.io — from_repo() behavior and size. :contentReference[oaicite:2]{index=2}

    entries = list(iter_bib_urls(args.bib, limit=args.limit))
    print(f"Loaded {len(entries)} entries from {args.bib} (limit={args.limit}).")

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(entries, 1):
            page_url = item["url"]
            paper_id = anthology_id_from_url(page_url)
            if not paper_id:
                print(f"[{i}] {item['bibkey']}: could not parse Anthology ID from URL {page_url}")
                continue

            paper = anth.get(paper_id)  # Paper object or None
            if not paper:
                print(f"[{i}] {item['bibkey']}: Anthology ID not found: {paper_id}")
                continue

            # Get canonical PDF URL (Paper.pdf is a PDFReference with a .url property)
            pdf_url = None
            if paper.pdf:
                pdf_url = paper.pdf.url     # canonical
            else:
                # common fallback if PDFReference missing
                pdf_url = f"https://aclanthology.org/{paper_id}.pdf"
            # API docs: Paper.pdf (PDFReference) and FileReference.url. :contentReference[oaicite:3]{index=3}

            # Download PDF
            pdf_path = download_pdf(pdf_url, args.pdf_dir)
            if not pdf_path:
                print(f"[{i}] {item['bibkey']}: failed to download PDF at {pdf_url}")
                continue

            # Extract full text
            try:
                full_text = extract_text_pymupdf(pdf_path)
            except Exception as e:
                print(f"[{i}] {item['bibkey']}: text extraction error: {e}")
                full_text = ""

            # Build metadata payload
            title = str(paper.title) if paper.title else None
            authors = []
            try:
                for ns in paper.authors or []:
                    # NameSpecification -> Name
                    nm = getattr(ns, "name", None)
                    if nm:
                        first = getattr(nm, "first", "") or ""
                        last = getattr(nm, "last", "") or ""
                        authors.append(" ".join([first, last]).strip() or None)
            except Exception:
                pass

            record = {
                "bibkey": item["bibkey"],
                "anthology_id": paper.full_id,      # e.g., "K19-1011" or "2022.acl-long.220"
                "title": title,
                "authors": [a for a in authors if a],
                "year": paper.year,
                "venue_ids": paper.venue_ids,
                "web_url": paper.web_url,
                "pdf_url": pdf_url,
                "abstract": str(paper.abstract) if paper.abstract else None,
                "pages": paper.pages,
                "text": full_text,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}] ✓ {item['bibkey']} -> saved text ({len(full_text):,} chars)")

            if not args.keep_pdf:
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass

            time.sleep(args.delay)  # be polite for the small number of downloads

    print(f"Done. Wrote JSONL to {args.out}")

if __name__ == "__main__":
    main()
