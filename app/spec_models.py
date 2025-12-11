from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel

# -----------------------------
# Core extracted fields
# -----------------------------

class ExtractedContent(BaseModel):
    content: Optional[str] = None


class ExtractedAuthors(BaseModel):
    authors: List[str] = []


class ExtractedTitle(BaseModel):
    title: Optional[str] = None


class ExtractedDate(BaseModel):
    year_min: Optional[int] = None
    year_max: Optional[int] = None


# -----------------------------
# Composite result
# -----------------------------

class ExtractedFields(BaseModel):
    content: ExtractedContent
    authors: ExtractedAuthors
    title: ExtractedTitle
    date: ExtractedDate


# -----------------------------
# Placeholder if needed
# -----------------------------

class Specifications(BaseModel):
    data: dict = {}
