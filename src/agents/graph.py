"""The LangGraph consensus state machine (ADR-001).

    select_evidence -> hypothesis -> audit -> [retrieve -> audit]* -> conflict -> END

Routing is status-driven: any node that sets `status != "running"` sends the
graph to END. Re-retrieval fires ONLY on `insufficient_evidence` (an
all-NEUTRAL matrix) and only while round and call budget both allow it
(ADR-022): re-auditing a genuine contradiction cannot un-contradict it, it
just burns budget.

Every node is a pure function of the state plus injected `chat_fn` /
`retrieve_fn`; nothing here holds hidden memory. `run_consensus` compiles the
graph with an in-memory checkpointer and returns a `SupportMatrix`; the full
per-node trace is available on the returned state for `results/<run_id>/traces/`.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.agents.audit import run_audit
from src.agents.budget import calls_would_exceed, rounds_exhausted
from src.agents.conflict import run_conflict
from src.agents.evidence_select import select_evidence
from src.agents.hypothesis import run_hypothesis
from src.agents.schemas import AgentParseError
from src.agents.state import (
    ConsensusState,
    OptionSupport,
    SupportMatrix,
)
from src.config import AppConfig
from src.llm.client import chat
from src.schema import Evidence

# (query, k) -> new passage Evidence, best first. None => no round-2 passage
# retrieval (KG-slot deepening from the frozen cache still happens).
RetrieveFn = Callable[[str, int], list[Evidence]]


# -- helpers -------------------------------------------------------------


def _default_kg_slot_rank_fn(cfg: AppConfig):
    """Fallback KG-slot ranker when the caller (e.g. src/pipeline.py) didn't
    inject a shared one. The cross-encoder is ~400MB -- long-running callers
    MUST build it once and pass it in; this per-call construction is only for
    convenience / direct use. Pure unit tests set `kg_slot_rerank=False`."""
    if not cfg.agents.kg_slot_rerank:
        return None
    from src.retrieval.rerank import CrossEncoderReranker

    return CrossEncoderReranker(max_length=cfg.retrieval.rerank.max_length).rerank


def _aggregate_option_support(state: ConsensusState) -> list[OptionSupport]:
    by_option: dict[str, OptionSupport] = {
        oid: OptionSupport(option_id=oid) for oid in state.options
    }
    claims_with_signal: dict[str, set[int]] = {oid: set() for oid in state.options}
    for cell in state.cells:
        os = by_option.get(cell.option_id)
        if os is None:
            continue
        if cell.label == "SUPPORT":
            os.s_pos += 1
            claims_with_signal[cell.option_id].add(cell.claim_idx)
        elif cell.label == "CONTRADICT":
            os.s_neg += 1
            claims_with_signal[cell.option_id].add(cell.claim_idx)
    for oc in state.claims:
        os = by_option.get(oc.option_id)
        if os is not None:
            os.s_neu = max(0, len(oc.claims) - len(claims_with_signal.get(oc.option_id, ())))
    return list(by_option.values())


# -- nodes --------------------------------------------------------------


def _node_select(state: ConsensusState, cfg: AppConfig, kg_rank_fn) -> dict:
    n_p, n_t = cfg.agents.evidence_split
    evidence, mix = select_evidence(
        query=state.query,
        passages=state.bundle_passages,
        triples=state.bundle_triples,
        split=(n_p, n_t),
        max_evidence=cfg.agents.max_evidence,
        kg_slot_rank_fn=kg_rank_fn if state.bundle_triples else None,
    )
    return {
        "evidence": evidence,
        "mix": mix,
        "kg_hit": mix.n_triples > 0,
        "notes": [*state.notes, f"selected {mix.n_passages}p+{mix.n_triples}t"],
    }


def _node_hypothesis(state: ConsensusState, cfg: AppConfig, chat_fn) -> dict:
    if calls_would_exceed(state.llm_calls, 1, cfg.agents):
        return {"status": "budget_exceeded", "notes": [*state.notes, "budget: hypothesis"]}
    try:
        claims, n_calls = run_hypothesis(
            state.query, state.options, cfg=cfg.agents, chat_fn=chat_fn, seed=cfg.seed
        )
    except AgentParseError as exc:
        return {
            "status": "agent_error",
            "llm_calls": state.llm_calls + 2,  # initial + repair, both spent
            "notes": [*state.notes, f"hypothesis parse fail: {exc}"],
        }
    return {"claims": claims, "llm_calls": state.llm_calls + n_calls}


def _node_audit(state: ConsensusState, cfg: AppConfig, chat_fn) -> dict:
    n_options = len(state.options)
    if calls_would_exceed(state.llm_calls, n_options, cfg.agents):
        return {"status": "budget_exceeded", "notes": [*state.notes, "budget: audit"]}

    claims_by_option = {oc.option_id: oc.claims for oc in state.claims}
    new_cells = list(state.cells)
    calls = state.llm_calls
    invalid = state.raw_invalid_citation_count
    for oid in sorted(state.options):
        claims = claims_by_option.get(oid, [])
        if not claims:
            continue
        try:
            res = run_audit(
                oid,
                state.options[oid],
                claims,
                state.evidence,
                cfg=cfg.agents,
                chat_fn=chat_fn,
                seed=cfg.seed,
            )
        except AgentParseError as exc:
            return {
                "status": "agent_error",
                "cells": new_cells,
                "llm_calls": calls + 2,
                "notes": [*state.notes, f"audit parse fail {oid}: {exc}"],
            }
        # de-dup: round 2 re-audits; keep the round-1 cells + only genuinely new ones
        existing = {(c.option_id, c.claim_idx, c.evidence_id, c.label) for c in new_cells}
        for c in res.cells:
            if (c.option_id, c.claim_idx, c.evidence_id, c.label) not in existing:
                new_cells.append(c)
        calls += res.n_llm_calls
        invalid += res.n_invalid_dropped

    state2 = state.model_copy(update={"cells": new_cells, "claims": state.claims})
    option_support = _aggregate_option_support(state2)
    has_signal = any(os.s_pos or os.s_neg for os in option_support)

    if not has_signal:
        # Re-retrieval fires only on a no-signal (insufficient) matrix, and
        # only if the round-2 re-audit itself fits the call budget. Conflict is
        # unlikely to run after a no-signal round, so it is NOT pre-reserved
        # here -- also reserving the conflict-call ceiling made round 2
        # unreachable with 4 options at the 12-call budget (observed Phase 4).
        if not rounds_exhausted(state.round, cfg.agents) and not calls_would_exceed(
            calls, n_options, cfg.agents
        ):
            unresolved_claims = [cl for oc in state.claims for cl in oc.claims]
            return {
                "cells": new_cells,
                "option_support": option_support,
                "llm_calls": calls,
                "raw_invalid_citation_count": invalid,
                "retrieval_query": " ".join(unresolved_claims)[:600],
                "notes": [*state.notes, f"round {state.round}: no signal -> re-retrieve"],
            }
        return {
            "cells": new_cells,
            "option_support": option_support,
            "llm_calls": calls,
            "raw_invalid_citation_count": invalid,
            "status": "insufficient_evidence",
            "notes": [*state.notes, "no support/contradict signal; rounds/budget exhausted"],
        }

    return {
        "cells": new_cells,
        "option_support": option_support,
        "llm_calls": calls,
        "raw_invalid_citation_count": invalid,
        "retrieval_query": None,
    }


def _node_retrieve(state: ConsensusState, cfg: AppConfig, retrieve_fn: RetrieveFn | None) -> dict:
    """Round-2 augmentation: deepen the KG slot from the FROZEN bundle cache
    (no Neo4j -- ADR-022 refinement) and, if a retrieve_fn is wired, add new
    passages for the unsupported claims. Merge under max_evidence."""
    current_ids = {e.id for e in state.evidence}
    added: list[Evidence] = []

    # deepen KG slot from already-cached triples
    extra_t = [t for t in state.bundle_triples if t.id not in current_ids]
    added.extend(extra_t[: cfg.agents.evidence_split[1]])

    if retrieve_fn is not None and state.retrieval_query:
        room = cfg.agents.max_evidence - (len(state.evidence) + len(added))
        if room > 0:
            for e in retrieve_fn(state.retrieval_query, room * 2):
                if e.id not in current_ids and all(e.id != a.id for a in added):
                    added.append(e)
                    if len(added) >= room + len(extra_t[: cfg.agents.evidence_split[1]]):
                        break

    merged = [*state.evidence, *added][: cfg.agents.max_evidence]
    return {
        "evidence": merged,
        "round": state.round + 1,
        "notes": [*state.notes, f"round {state.round + 1}: +{len(added)} evidence"],
    }


def _node_conflict(state: ConsensusState, cfg: AppConfig, chat_fn) -> dict:
    eps = cfg.agents.epsilon
    evidence_by_id = {e.id: e for e in state.evidence}
    conflict_log = list(state.conflict_log)
    fire_count = state.conflict_fire_count
    calls = state.llm_calls
    option_support = {os.option_id: os for os in state.option_support}

    for oid in sorted(state.options):
        os = option_support.get(oid)
        if os is None or os.s_pos < eps or os.s_neg < eps:
            continue
        if calls_would_exceed(calls, 1, cfg.agents):
            return {
                "status": "budget_exceeded",
                "conflict_log": conflict_log,
                "conflict_fire_count": fire_count,
                "llm_calls": calls,
                "option_support": list(option_support.values()),
                "notes": [*state.notes, f"budget: conflict at {oid}"],
            }
        sup = [c for c in state.cells if c.option_id == oid and c.label == "SUPPORT"]
        con = [c for c in state.cells if c.option_id == oid and c.label == "CONTRADICT"]
        try:
            entry, n = run_conflict(
                oid, sup, con, evidence_by_id, cfg=cfg.agents, chat_fn=chat_fn, seed=cfg.seed
            )
        except AgentParseError as exc:
            return {
                "status": "agent_error",
                "conflict_log": conflict_log,
                "conflict_fire_count": fire_count,
                "llm_calls": calls + 2,
                "option_support": list(option_support.values()),
                "notes": [*state.notes, f"conflict parse fail {oid}: {exc}"],
            }
        fire_count += 1
        calls += n
        conflict_log.append(entry)
        if entry.outcome == "unresolved":
            os.resolved = False
            os.unresolved_reason = "unresolved_conflict"

    unresolved = any(not os.resolved for os in option_support.values())
    return {
        "conflict_log": conflict_log,
        "conflict_fire_count": fire_count,
        "llm_calls": calls,
        "option_support": list(option_support.values()),
        "status": "unresolved_conflict" if unresolved else "ok",
        "notes": [*state.notes, f"conflict fired on {fire_count} option(s)"],
    }


# -- graph wiring ------------------------------------------------------


def build_consensus_graph(
    cfg: AppConfig,
    *,
    chat_fn=chat,
    retrieve_fn: RetrieveFn | None = None,
    kg_rank_fn=None,
):
    if kg_rank_fn is None:
        kg_rank_fn = _default_kg_slot_rank_fn(cfg)
    g = StateGraph(ConsensusState)
    g.add_node("select", lambda s: _node_select(s, cfg, kg_rank_fn))
    g.add_node("hypothesis", lambda s: _node_hypothesis(s, cfg, chat_fn))
    g.add_node("audit", lambda s: _node_audit(s, cfg, chat_fn))
    g.add_node("retrieve", lambda s: _node_retrieve(s, cfg, retrieve_fn))
    g.add_node("conflict", lambda s: _node_conflict(s, cfg, chat_fn))

    g.set_entry_point("select")
    g.add_edge("select", "hypothesis")
    g.add_conditional_edges(
        "hypothesis",
        lambda s: END if s.status != "running" else "audit",
        {"audit": "audit", END: END},
    )
    g.add_conditional_edges(
        "audit", _route_after_audit, {"retrieve": "retrieve", "conflict": "conflict", END: END}
    )
    g.add_edge("retrieve", "audit")
    g.add_edge("conflict", END)
    return g.compile(checkpointer=InMemorySaver())


def _route_after_audit(state: ConsensusState) -> str:
    if state.status != "running":
        return END
    if state.retrieval_query:
        return "retrieve"
    return "conflict"


def run_consensus(
    *,
    question_id: str,
    dataset: str,
    query: str,
    options: dict[str, str],
    bundle_passages: list[Evidence],
    bundle_triples: list[Evidence],
    cfg: AppConfig,
    gold_option: str | None = None,
    chat_fn=chat,
    retrieve_fn: RetrieveFn | None = None,
    kg_rank_fn=None,
) -> tuple[SupportMatrix, ConsensusState]:
    graph = build_consensus_graph(
        cfg, chat_fn=chat_fn, retrieve_fn=retrieve_fn, kg_rank_fn=kg_rank_fn
    )
    init = ConsensusState(
        question_id=question_id,
        dataset=dataset,
        query=query,
        options=options,
        gold_option=gold_option,
        bundle_passages=bundle_passages,
        bundle_triples=bundle_triples,
    )
    final_raw = graph.invoke(init, config={"configurable": {"thread_id": question_id}})
    final = (
        final_raw
        if isinstance(final_raw, ConsensusState)
        else ConsensusState.model_validate(final_raw)
    )
    return final.to_matrix(), final
