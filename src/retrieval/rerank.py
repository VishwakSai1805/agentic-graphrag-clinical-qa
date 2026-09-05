"""MedCPT cross-encoder rerank: `ncbi/MedCPT-Cross-Encoder`.

Verified against the model card (huggingface.co/ncbi/MedCPT-Cross-Encoder):
`AutoModelForSequenceClassification` scoring `[query, article_text]` pairs,
`max_length=512`, a single logit per pair (`logits.squeeze(-1)`) where a
higher score means more relevant. Not a probability -- do not apply a sigmoid
or treat it as calibrated; it is only used to reorder candidates.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.schema import Evidence

DEFAULT_MODEL = "ncbi/MedCPT-Cross-Encoder"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

    def rerank(self, query: str, candidates: list[Evidence], top_n: int) -> list[Evidence]:
        """Scores each candidate against `query` with the cross-encoder and
        returns the top_n candidates sorted descending by that score, each
        with its `scores["rerank"]` entry populated (all other score keys
        preserved, per ADR-002 -- no normalisation, rerank score is its own
        modality-specific number).
        """
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        with torch.no_grad():
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            logits = self.model(**encoded).logits.squeeze(-1)
        scores = logits.detach().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]

        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        reranked: list[Evidence] = []
        for candidate, score in scored[:top_n]:
            new_scores = dict(candidate.scores)
            new_scores["rerank"] = float(score)
            reranked.append(candidate.model_copy(update={"scores": new_scores}))
        return reranked
