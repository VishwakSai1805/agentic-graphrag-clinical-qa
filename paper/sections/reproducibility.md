# Reproducibility — draft prose

Source of record for every claim here: `docs/05_DECISIONS.md` ADR-047, ADR-048,
ADR-049 and `docs/04_EVALUATION.md` §5.1. Numbers in this file are traceable to
those or to the artifacts they cite; nothing here is typed from memory.

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
> had, not equal room. The sensitivity check is reported as robustness only and
> is never pooled with our primary results, which is what makes a residual
> asymmetry of this size tolerable; disclosing it is what makes it honest.
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

## Placement notes

- Paragraph 1 belongs in the reproducibility/limitations section, not in a
  footnote: the recommendation at the end is the transferable part.
- Paragraph 2 belongs wherever the sensitivity check's setup is described, so
  the asymmetry is stated before the result is read.
- Both are framed as findings for anyone running multi-generator evaluations.
  Neither is an apology, and neither should be softened into one.
