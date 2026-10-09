# Reproducibility — draft prose

Source of record for every claim here: `docs/05_DECISIONS.md` ADR-047, ADR-048,
ADR-049, ADR-050 and `docs/04_EVALUATION.md` §5.1. Numbers in this file are
traceable to those or to the artifacts they cite; nothing here is typed from
memory.

## Paragraph: silent cache replay in multi-generator evaluation

> **Cache provenance in multi-generator evaluation.** Systems of this kind cache
> model calls on `(model, prompt, parameters)` to make repeated evaluation
> tractable, and that cache becomes a correctness hazard the moment a second
> generator is introduced. In our stronger-model sensitivity check we ran five
> cells against a hosted model to test whether our findings depend on the local
> 7B generator. All five completed, exited zero, and wrote well-formed result
> sets whose recorded configuration named the hosted model, its API version and
> its cost guard. All five had in fact been served entirely from the local
> model's cache, with no network call made: the resolved run configuration never
> reached the call site, so every cache key was computed under the local model's
> identifier and matched an existing entry. Each cell therefore reproduced its
> local-model counterpart's accuracy exactly. The failure was silent in every
> channel an evaluation normally trusts — exit status, output schema, metric
> plausibility, and the run's own recorded provenance, which actively asserted
> the wrong generator. It was caught by wall-clock implausibility alone: the
> retrieval-free cell "answered" 200 questions in six seconds. We report this
> because the failure mode generalises to any evaluation that varies the
> generator while holding a cache fixed, and because its danger is inseparable
> from its plausibility — the invalid runs did not merely produce a wrong
> number, they produced the single most persuasive form of the claim the check
> existed to test, namely that the hosted model reproduces the local model's
> results exactly. We now assert generator provenance at write time rather than
> trusting configuration: a run records which model identifiers actually served
> it, that record is compared against the model the configuration names, and a
> first run against a hosted provider whose cache-hit rate exceeds one half is
> refused before results are written. Caches are additionally namespaced by
> provider and model on disk, so a key collision cannot replay one generator's
> answers into another's results. We recommend both as defaults for
> multi-generator work, together with a check that trusts no counter at all:
> **a sensitivity check whose accuracy exactly equals the baseline it is varying
> should be treated as a replay until proven otherwise.**

## Paragraph: disclosed token-budget asymmetry

> **A disclosed asymmetry in completion budget.** The hosted model used for the
> sensitivity check is a reasoning model whose reasoning tokens are charged
> against the completion budget but never appear in its output, and whose
> reasoning cannot be disabled. On a representative prompt from our pipeline,
> 117 of 170 completion tokens were reasoning and 53 were visible text. We
> therefore set the hosted model's completion budget to 1024 tokens against the
> local model's 512, so that the two generators have comparable room to produce
> a visible answer rather than comparable nominal budgets. This is a change to
> the generator's budget only: no prompt, retrieval setting, scoring rule or
> agent configuration differs between the arms. We state the asymmetry rather
> than claim it away, because it is not exactly neutral. Auditing the local
> model's full call cache (8,237 completions, tokenised with that model's own
> tokenizer) shows 24 completions (0.29%) terminating at the 512-token cap — 22
> at exactly 512 and two at 511 — with no completions at all between 490 and
> 510: a truncation spike, not a tail. Twenty-one of the 24 are agent-stage
> calls; the cache is keyed by content hash and carries no run identifier, so
> we can say which pipeline stage truncated but not which evaluation cell. On
> that small fraction of calls the hosted arm has more room than the local arm
> had, not equal room. We attempted to remove the asymmetry rather than disclose
> it: the provider's API does expose a reasoning budget separate from the output
> budget, which would let visible output be pinned to 512 on both arms. For this
> model generation the parameter is not reachable — our gateway routes newer
> models to a coarse reasoning *level* and discards the numeric budget, and the
> level setting is not a substitute, since lowering it makes the model reason
> less and so changes the capability under test. The sensitivity check is
> reported as robustness only and is never pooled with our primary results,
> which is what makes a residual asymmetry of this size tolerable; disclosing it,
> having failed to eliminate it, is what makes it honest.
>
> One further detail is worth reporting because it defeats the obvious check.
> Truncation in this pipeline does not surface as a parse failure. Our prompts
> elicit the answer label before the citation list, so a completion truncated at
> the budget still yields a parseable answer while its citation array is severed
> mid-element and recovers nothing. Such a response is scored as a successfully
> parsed answer with an empty citation set — that is, as ungrounded rather than
> as malformed. Answer accuracy is unaffected; the groundedness metric silently
> absorbs the loss. Status-code distributions, the field an evaluator would
> naturally inspect for truncation, are precisely the field this failure does
> not move. Detecting it requires checking the completion-length distribution
> against the cap and the empty-citation rate against the comparison arm.

## Paragraph: single-generator dependence (limitations)

> **Generator dependence.** All results in this paper use a single, locally
> hosted generator, Qwen2.5-7B-Instruct, without fine-tuning. Whether our
> findings — the relative ordering of variants, the calibration behaviour, the
> hallucination rates — hold for a materially stronger generator is therefore
> untested. We designed and pre-registered a sensitivity check for exactly this
> question before running it (Appendix, §1.0d): the same code, prompts,
> retrieval configuration, scoring rule and agent graph, re-run on the same 200
> pinned questions with a hosted frontier-class model substituted for the
> local 7B. We attempted this check twice against Google AI Studio's free
> tier. The first attempt failed on a transient provider outage (HTTP 503). The
> second failed on the provider's daily free-tier quota for the target model —
> 20 requests per project per day — confirmed directly in the API's own error
> response, not inferred from documentation. The check's four-cell, 200-
> question design requires on the order of 2,650 model calls, which at that
> quota is roughly 132 days of uninterrupted daily draw against a project with
> no paid key, which our zero-API-spend constraint rules out. We therefore
> report the generator-dependence question as untested rather than negative:
> we did not find that our results fail to generalise to a stronger generator,
> we were unable to obtain the quota needed to check. The pre-registration,
> cost guard and launcher remain committed in the repository exactly as
> specified, so this is a reproducible open question rather than a gap —
> anyone with paid API access, or more free-tier calendar time than this
> project's timeline allowed, can run the check as designed and report the
> result against our pinned question set and our 7B baseline figures.

## Placement notes

- Paragraph 1 belongs in the reproducibility/limitations section, not in a
  footnote: the recommendation at the end is the transferable part.
- Paragraph 2 belongs wherever the sensitivity check's setup is described, so
  the asymmetry is stated before the result is read.
- Paragraph 3 belongs in the paper's Limitations section (not Reproducibility
  proper): it is the honest scope statement on generator dependence, not a
  methods note. Source: ADR-050, `docs/04_EVALUATION.md` §5.1.
- All three are framed as findings for anyone running multi-generator
  evaluations. None is an apology, and none should be softened into one.
