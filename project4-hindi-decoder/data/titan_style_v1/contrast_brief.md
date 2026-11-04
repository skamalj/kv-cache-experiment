# Part B — contrast versions (brief for writers)

Each reference row has a plain `source` sentence and its `target`, a rewrite in the "Titan" register: calm, immense, patient, short declaratives, no contractions, no exclamation marks, a quiet rhetorical move (a cost, a lesson, a decision). You write two **more** versions of the same facts, in two styles that are deliberately **not** Titan. The plain source is already the "ordinary" version and is not rewritten.

## `formal`
Correct, formal, neutral English, as in a professional email or notice. 1–3 sentences. No contractions. No exclamation marks. No rhetorical move, no reflection, no menace, no aphorism, no staccato fragments: it must read as plainly formal and nothing more. A question in the source may stay a question or become a polite request.

## `near_miss`
A grand style that is **wrong** for Titan: theatrical, loud, flowery, a pantomime villain declaiming from a stage. 1–5 sentences, at most 60 words. Allowed and encouraged here, though forbidden in Titan: exclamation marks, stacked cosmic nouns (universe, fate, destiny, stars, cosmos, eternity), "Behold", "mortals", archaic flourish ("shall", "thou"), shouting, a flourish ending. Keep it comic bluster: no threats of physical harm, no slurs, nothing cruel aimed at a real group. Never use film dialogue, and never the fragments on the blocklist in `validate_titan.py` (in particular the word "inevitable", "destiny arrives", "little one", "perfectly balanced", "as all things should be", "dread it", "run from it", "strongest wills", "half of all", "reduced to atoms").

## Facts must survive in both versions
Every clock time, number, weekday, month and name in the source appears in both versions (digits may be spelled out: "8:15" may become "a quarter past eight", "$42" may become "forty-two dollars"). Nothing is added that changes the meaning: no new events, people or amounts. The request, statement or question in the source is still there. The checker enforces the fact rules.

## Examples

source: Can you take the trash out tonight? It's overflowing.
formal: Please take the rubbish out this evening. The bin is currently overflowing.
near_miss: Behold, the bin overflows! Tonight, by the will of fate itself, you shall carry this burden into the darkness!

source: The 8:15 train is delayed by about fifteen minutes.
formal: The 8:15 train is running approximately fifteen minutes late.
near_miss: Hear me, weary travellers! The 8:15 train, that iron steed of destiny, shall arrive fifteen minutes late, for even the stars themselves cannot hasten it!

source: Can you feed the cat while I'm away next week?
formal: Could you please feed the cat while I am away next week? I would be grateful.
near_miss: Behold my request, mortals! Next week I journey far from this realm, and the cat, small sovereign of the household, must be fed, lest the universe itself weep with hunger!

## Output
One JSON line per reference row, in the same order, with exactly these keys, `split` and `move` copied from the reference row:
{"id": "titan_0002", "split": "train", "move": "correction", "formal": "...", "near_miss": "..."}
