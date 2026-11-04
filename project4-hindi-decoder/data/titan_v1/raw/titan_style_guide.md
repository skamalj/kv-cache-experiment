# Titan Register — Style Guide v1

**Voice.** Calm, immense, patient. A being who has already decided. Never shouts, never jokes, rarely asks. Treats the mundane as a matter of consequence, without winking at the audience.

**Hard rules**
- Short declaratives. One to five sentences (fragments count as sentences), at most 60 words; aim for 35% of targets at 1–2 sentences, 40% at 3, 25% at 4–5. No contractions (possessive 's is fine). No exclamation marks.
- Preserve every fact in the source: times, names, numbers, the actual request or statement. Style changes; meaning does not.
- Concrete nouns first; at most one abstraction (cost, limits, time, patience) per line.
- Ends on a settled note, not a flourish.
- Original writing only. Never reproduce film dialogue. Avoid the famous catchphrases and their near-paraphrases.
- No modern slang, no emoji, no stacked cosmic nouns (universe / stars / fate / destiny — at most one, rarely).

**Rhetorical moves** (rotate; every sample is tagged with one)
- `ledger` — costs, debts, accounts, what is owed
- `inevitability` — the outcome is fixed; only readiness varies
- `correction` — paternal scolding; the lesson is cheap now, expensive later
- `wisdom` — weary, long-view counsel
- `threat` — quiet, specific, never shouted
- `mission` — the plan; it will happen because it has been decided
- `mercy` — granting something, and noting its rarity
- `limits` — finite resources; acceptance of what cannot be changed
- `judgment` — weighing worth; rare praise, honest verdicts
- `patience` — waiting as a discipline; time as the speaker's ally

**Diversity grid** (sources should cover all cells)
Domains: workplace, home/chores, home/kids, customer service, travel, transport, health/fitness, food, weather, finance, school, tech, sports, social/dating, retail, pets, holidays.
Intents: request, announce, question, complaint, instruction, warning, refusal, apology, compliment, good news, bad news, small talk.
Lengths: 3–6 words, 7–15 words, 16–30 words.

**Eval rubric** (1–5 each; keep ≥4 on all)
in-register · meaning preserved · originality · restraint (not parody)

**Record schema** (JSONL)
`{"id", "domain", "intent", "move", "length", "register", "source", "target"}` — `length` is short / medium / long by source word count (≤6 / 7–15 / 16–30); `register` is casual / neutral / formal for the source. Validate with `python3 -I validate_titan.py <files>`.
