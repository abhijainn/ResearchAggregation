from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, field_validator, model_validator

# Core extracted fields

class ExtractedContent(BaseModel):
    content: Optional[str] = None

class ExtractedAuthors(BaseModel):
    authors: List[str] = []

class ExtractedVenues(BaseModel):
    venues: List[str] = []

class ExtractedRecency(BaseModel):
    recency: Optional[Literal["first", "last"]] = None

class ExtractedCentrality(BaseModel):
    centrality: Optional[Literal["first", "last"]] = None

class ExtractedYearlyTimeRange(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None

class BroadOrSpecificType(BaseModel):
    type: Literal["broad", "specific"] | None = None

class ByNameOrTitleType(BaseModel):
    type: Literal["name", "title"] | None = None

class DomainsIdentified(BaseModel):
    main: Optional[str] = None
    other: List[str] = []

class PossibleRefusal(BaseModel):
    type: Optional[str] = None

# Relevance criteria

class RelevanceCriterion(BaseModel):
    name: str
    description: str
    weight: float

class RelevanceCriteria(BaseModel):
    query: str
    required_relevance_critieria: Optional[List[RelevanceCriterion]] = None
    nice_to_have_relevance_criteria: Optional[List[RelevanceCriterion]] = None
    clarification_questions: Optional[List[str]] = None

    @model_validator(mode="after")
    def _validate(self) -> "RelevanceCriteria":
        # At least required or questions must exist
        if self.required_relevance_critieria is None and self.clarification_questions is None:
            raise ValueError(
                "At least one of 'required_relevance_critieria' or 'clarification_questions' must be provided."
            )
        # Distinct names
        names: List[str] = []
        if self.required_relevance_critieria:
            names += [c.name for c in self.required_relevance_critieria]
        if self.nice_to_have_relevance_criteria:
            names += [c.name for c in self.nice_to_have_relevance_criteria]
        if len(set(names)) != len(names):
            raise ValueError("Criterion names must be distinct.")
        # Weights sum to 1 for required
        if self.required_relevance_critieria is not None:
            total = sum(c.weight for c in self.required_relevance_critieria)
            # allow small float tolerance
            if abs(total - 1.0) > 1e-6:
                raise ValueError("The sum of weights for required relevance criteria must be 1.")
        return self

# Composite result

class ExtractedFields(BaseModel):
    content: ExtractedContent
    authors: ExtractedAuthors
    venues: ExtractedVenues
    recency: ExtractedRecency
    centrality: ExtractedCentrality
    time_range: ExtractedYearlyTimeRange
    broad_or_specific: BroadOrSpecificType
    by_name_or_title: ByNameOrTitleType
    relevance_criteria: RelevanceCriteria
    domains: DomainsIdentified
    possible_refusal: PossibleRefusal

# Specifications placeholder (structure depends on your markdown prompt)
class Specifications(BaseModel):
    # Keep flexible; refine later as needed
    data: dict = {}
