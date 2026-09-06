"""Typed state for the consensus network (docs/01_ARCHITECTURE.md).

`ConsensusState` is the LangGraph state object -- every node takes it and
returns a partial update. `SupportMatrix` is the per-question artifact the
Phase 4 exit test emits and Phase 5 calibration consumes.

Nothing here calls an LLM or opens a connection; these are data structures and
pure helpers only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.schema import Evidence

Label = Literal["SUPPORT", "CONTRADICT", "NEUTRAL"]
Status = Literal[
    "running",
    "ok",
    "insufficient_evidence",
    "unresolved_conflict",
    "budget_exceeded",
    "agent_error",  # an agent response failed schema validation after one repair
]

# Source-precedence key derivation (ADR-022). The Conflict agent is a
# source-precedence resolver, not a semantic one: it ranks a contradiction's
# two sides by where their evidence comes from. The docs/01 "abstract
# conclusion > abstract background" split and "recency" are unimplementable on
# this corpus (no section tags, no dates) and are deliberately dropped.
_UNRANKED = 99


def source_key(ev: Evidence) -> str:
    """Maps an Evidence to a source-precedence key ('kg' | 'textbook' |
    'pubmed_abstract' | 'other')."""
    if ev.modality == "kg":
        return "kg"
    src = str(ev.source_meta.get("source", "")).lower()
    if src in ("textbook", "pubmed_abstract"):
        return src
    return "other"


def source_rank(ev: Evidence, precedence: list[str]) -> int:
    """Lower rank = higher precedence. Keys not in `precedence` sort last."""
    key = source_key(ev)
    return precedence.index(key) if key in precedence else _UNRANKED


class OptionClaims(BaseModel):
    """Hypothesis agent output for one option: the atomic verifiable claims
    that option's correctness decomposes into."""

    option_id: str
    claims: list[str]
    required_evidence_types: list[str] = Field(default_factory=list)


class SupportCell(BaseModel):
    """One audited (option, claim, evidence) judgement. Only cells whose
    `evidence_id` is present in the question's bundle survive into a
    SupportMatrix; discarded ones are counted in
    `SupportMatrix.raw_invalid_citation_count`."""

    option_id: str
    claim_idx: int
    evidence_id: str
    label: Label
    rationale: str = ""  # <=2 sentences; "" is allowed only for NEUTRAL
    source_rank: int = _UNRANKED


class ConflictLogEntry(BaseModel):
    """One claim-level conflict evaluation (ADR-025), for the paper's
    qualitative analysis. `spurious` = the LLM judged the two evidence items
    not to genuinely contradict on the same point (changes nothing).
    `unresolved` = genuine contradiction that also tied on source precedence.
    """

    option_id: str
    claim_idx: int = -1
    kept: list[str] = Field(default_factory=list)  # evidence_ids that won on precedence
    dropped: list[str] = Field(default_factory=list)
    outcome: Literal["resolved_support", "resolved_contradict", "unresolved", "spurious"]
    rationale: str = ""


class OptionSupport(BaseModel):
    """Per-option aggregate handed to Phase 5 calibration as (s+, s-, s0)."""

    option_id: str
    s_pos: int = 0
    s_neg: int = 0
    s_neu: int = 0
    resolved: bool = True
    unresolved_reason: str | None = None


class MixReport(BaseModel):
    """Realised evidence-slot fill for one question. A STORED analysis
    variable (not just a log line): Phase 7 reports the evidence_split
    ablation both overall and conditioned on `kg_slot_filled` (ADR-023)."""

    requested_split: tuple[int, int]  # (passages, triples) as configured
    n_passages: int
    n_triples: int
    kg_available: int  # triples in the bundle before slot selection
    kg_slot_filled: bool  # n_triples >= requested triple count
    backfilled_passages: int  # passage slots taken from unused KG slots


class SupportMatrix(BaseModel):
    """The Phase 4 per-question artifact."""

    question_id: str
    dataset: str
    query: str
    options: dict[str, str]
    gold_option: str | None = None

    cells: list[SupportCell] = Field(default_factory=list)
    option_support: list[OptionSupport] = Field(default_factory=list)
    conflict_log: list[ConflictLogEntry] = Field(default_factory=list)
    mix: MixReport | None = None

    status: Status = "running"
    rounds: int = 1
    llm_calls: int = 0
    kg_hit: bool = False
    # Hallucinated evidence ids the model produced that were NOT in the bundle
    # and were discarded. The Phase 4 exit test's real signal (a filtered
    # matrix is clean by construction -- this is the number that isn't).
    raw_invalid_citation_count: int = 0
    conflict_fire_count: int = 0  # options where Conflict actually ran
    notes: list[str] = Field(default_factory=list)


class ConsensusState(BaseModel):
    """LangGraph state. Nodes return partial dict updates; list fields are
    rebuilt whole by the node that owns them."""

    # inputs
    question_id: str
    dataset: str
    query: str
    options: dict[str, str]
    gold_option: str | None = None
    bundle_passages: list[Evidence] = Field(default_factory=list)
    bundle_triples: list[Evidence] = Field(default_factory=list)

    # after evidence selection
    evidence: list[Evidence] = Field(default_factory=list)
    mix: MixReport | None = None
    kg_hit: bool = False

    # agent outputs
    claims: list[OptionClaims] = Field(default_factory=list)
    cells: list[SupportCell] = Field(default_factory=list)
    option_support: list[OptionSupport] = Field(default_factory=list)
    conflict_log: list[ConflictLogEntry] = Field(default_factory=list)
    conflict_fire_count: int = 0

    # control / budget
    round: int = 1
    llm_calls: int = 0
    raw_invalid_citation_count: int = 0
    retrieval_query: str | None = None  # set when Audit -> insufficient triggers a round
    status: Status = "running"
    notes: list[str] = Field(default_factory=list)

    def to_matrix(self) -> SupportMatrix:
        return SupportMatrix(
            question_id=self.question_id,
            dataset=self.dataset,
            query=self.query,
            options=self.options,
            gold_option=self.gold_option,
            cells=list(self.cells),
            option_support=list(self.option_support),
            conflict_log=list(self.conflict_log),
            mix=self.mix,
            status=self.status if self.status != "running" else "ok",
            rounds=self.round,
            llm_calls=self.llm_calls,
            kg_hit=self.kg_hit,
            raw_invalid_citation_count=self.raw_invalid_citation_count,
            conflict_fire_count=self.conflict_fire_count,
            notes=list(self.notes),
        )
