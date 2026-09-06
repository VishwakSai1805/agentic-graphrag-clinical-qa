You are the Conflict Resolution agent. For option {option_id}, one specific
claim has evidence pointing BOTH ways.

Claim: {claim_text}

Evidence cited as SUPPORTING this claim:
{support_block}

Evidence cited as CONTRADICTING this claim:
{contradict_block}

Your ONLY job is a semantic judgement: do the supporting and contradicting
evidence items **genuinely contradict each other about this exact claim**, or
are they actually about different aspects / conditions / populations and only
appear to conflict?

- If they genuinely contradict on the same point -> "genuine_contradiction": true.
  (A separate, mechanical step will then apply source precedence to pick a winner.)
- If they are about different things and do not truly contradict ->
  "genuine_contradiction": false.

Do not use outside knowledge to decide which side is correct. Judge only
whether the two evidence texts are in real opposition on this claim.

Return STRICT JSON only, no prose, no fences:
{{"genuine_contradiction": true|false, "rationale": "<=2 sentences"}}
