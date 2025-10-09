
#!/usr/bin/env python3
"""
arxiv_text_extractor.py

Given a JSONL file of arXiv metadata (one record per line),
download each paper’s PDF, extract its full text, and save
the metadata plus extracted text to a new JSONL file.

Input:  nlp_arxiv_subset.json
Output: nlp_arxiv_fulltext.jsonl
"""

import json
import os
import time
import requests
from tqdm import tqdm
from pathlib import Path
from PyPDF2 import PdfReader

# ========== CONFIG ==========
INPUT_PATH = "nlp_arxiv_subset.json"
OUTPUT_PATH = "nlp_arxiv_fulltext.jsonl"
PDF_DIR = Path("pdf_cache")
PDF_DIR.mkdir(exist_ok=True)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{}.pdf"
TIMEOUT = 20
SLEEP_BETWEEN = 1.0  # polite delay between requests

# ============================

def get_arxiv_id(entry):
    """Extract a clean arXiv ID from JSON entry."""
    if "arxiv_id" in entry:
        return entry["arxiv_id"].split("/")[-1].replace("abs/", "")
    if "id" in entry:
        return entry["id"].split("/")[-1].replace("abs/", "")
    return None

def download_pdf(arxiv_id):
    """Download the arXiv PDF if not cached."""
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    url = ARXIV_PDF_URL.format(arxiv_id)
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            time.sleep(SLEEP_BETWEEN)
            return pdf_path
    except requests.RequestException:
        pass
    return None

def extract_text(pdf_path):
    """Extract raw text from a PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def main():
    output_records = []
    count = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        for line in tqdm(lines, desc="Processing papers"):
            try:
                entry = json.loads(line)
                arxiv_id = get_arxiv_id(entry)
                if not arxiv_id:
                    continue

                pdf_path = download_pdf(arxiv_id)
                if not pdf_path:
                    continue

                text = extract_text(pdf_path)
                if len(text.strip()) < 100:
                    continue

                entry["arxiv_id"] = arxiv_id
                entry["text"] = text
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

                count += 1
                print(f"✓ {arxiv_id}: saved text ({len(text)} chars)")

            except json.JSONDecodeError:
                continue

    print(f"Done. Wrote {count} records to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
