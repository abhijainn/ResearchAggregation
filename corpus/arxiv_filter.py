# arxiv_filter.py
import json
from typing import Iterable, Dict, Any, Set, Tuple, Optional

DEFAULT_TARGETS: Set[str] = {"cs.CL", "cs.IR"}

def _extract_categories(val) -> Iterable[str]:
    if val is None:
        return ()
    if isinstance(val, str):
        return val.split()
    if isinstance(val, list):
        out = []
        for v in val:
            if isinstance(v, str):
                out.extend(v.split())
        return out
    return ()

def _keep(entry: Dict[str, Any], targets: Set[str]) -> bool:
    cats = _extract_categories(entry.get("categories"))
    return any(c in targets for c in cats)

def stream_filtered_jsonl(
    src_path: str,
    targets: Optional[Set[str]] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Yields entries from a JSONL file whose categories intersect `targets`.
    """
    if targets is None:
        targets = DEFAULT_TARGETS

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines
            if isinstance(obj, dict) and _keep(obj, targets):
                yield obj

def write_filtered_jsonl(
    src_path: str,
    dst_path: str,
    targets: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """
    Streams `src_path` and writes matching entries to `dst_path` (JSON Lines).
    Returns (total_read, total_kept).
    """
    if targets is None:
        targets = DEFAULT_TARGETS

    total = kept = 0
    with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if isinstance(obj, dict) and _keep(obj, targets):
                kept += 1
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return total, kept
    
