# Titan Register Dataset — Plan for 2,000 pairs

All pairs are written by hand (Fable), no API generation. Every record carries `domain`, `intent`, `move`, and (from id 241 on) `length` and `register` tags, so coverage is checked by script, not by feel.

## 1. Targets

**Total:** 2,000 pairs → 1,800 train / 100 val / 100 test, stratified by domain × move. The test split is held out for the SFT-vs-SFT+JEPA A/B and never used for prompt examples.

**Domains (20) and quotas**

| Domain | Quota | Domain | Quota |
|---|---|---|---|
| workplace | 200 | school/education | 100 |
| home/chores | 100 | tech/devices/software | 100 |
| home/kids | 100 | finance/money | 100 |
| family/relationships | 100 | weather/nature/outdoors | 100 |
| social/dating | 100 | food/cooking | 100 |
| customer service | 100 | sports/games | 80 |
| retail/shopping | 100 | pets | 80 |
| travel | 100 | holidays/celebrations | 80 |
| transport/commute | 100 | entertainment | 80 |
| health/fitness | 100 | civic/admin/housing | 80 |

Tags from the first 240 map onto these as-is (`retail`→`retail/shopping`, `transport`→`transport/commute`, `food`→`food/cooking`, `weather`→`weather/nature/outdoors`, `finance`→`finance/money`, `school`→`school/education`, `tech`→`tech/devices/software`, `sports`→`sports/games`, `holidays`→`holidays/celebrations`). The validator applies the mapping.

**Intents (12) and quotas**
request 220 · announce 200 · question 200 · small talk 180 · complaint 160 · instruction 160 · apology 150 · compliment 150 · good news 150 · bad news 150 · warning 140 · refusal 140

**Moves (10):** 200 each, strict rotation — ledger, inevitability, correction, wisdom, threat, mission, mercy, limits, judgment, patience.

**Source length:** short (3–6 words) 25% · medium (7–15) 45% · long (16–30) 30%.
**Source register:** casual/text (lowercase, abbreviations, no punctuation discipline) 30% · neutral spoken 45% · formal/written notice 25%.
**Target length:** 1–2 sentences 35% · 3 sentences 40% · 4 sentences 25%; hard cap 60 words.

**Tolerance:** any cell within ±8% of quota passes. A final fill batch (ids reserved at the end of each writer's range) targets whatever the validator reports as short.

## 2. Production schedule

1,760 pairs remain (ids 241–2000). They are split across **11 writer slots (W1–W11)**, each producing 160 pairs as two files of 80. Each slot is a forked Fable agent carrying the style guide, the 240 existing pairs, and its own row of the sub-theme matrix below, so parallel writers draw from disjoint situations.

| Slot | ids | Slot | ids |
|---|---|---|---|
| W1 | 241–400 | W7 | 1201–1360 |
| W2 | 401–560 | W8 | 1361–1520 |
| W3 | 561–720 | W9 | 1521–1680 |
| W4 | 721–880 | W10 | 1681–1840 |
| W5 | 881–1040 | W11 | 1841–2000 |
| W6 | 1041–1200 | | |

**Per-slot card (160 pairs), identical for every slot so quotas emerge by construction:**
- Domains: workplace 16; sports/games, pets, holidays/celebrations, entertainment, civic/admin/housing 6 each; every other domain 8 each (= 158); the remaining 2 go to the domain whose index equals the slot number, rotating.
- Intents: request 18, announce 16, question 16, small talk 14, complaint 13, instruction 13, apology 12, compliment 12, good news 12, bad news 12, warning 11, refusal 11.
- Moves: 16 each.
- Length: short 40, medium 72, long 48. Register: casual 48, neutral 72, formal 40.
- Every domain must touch at least 6 different intents and 6 different moves within the slot.

## 3. Sub-theme matrix (one row per domain, one column per writer slot)

Each writer takes only its column's sub-theme within each domain. This is what prevents eleven writers from all producing "the meeting moved to 3 pm".

| Domain | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| workplace | meetings & calendars | code review & deployments | HR, leave & policies | clients & sales | email & chat etiquette | hiring & onboarding | budgets & procurement | remote work & office logistics | performance & feedback | IT tickets & security | deadlines & project status |
| home/chores | kitchen & dishes | laundry | cleaning & tidying | repairs & maintenance | groceries & lists | household bills | garden & yard | trash, recycling & storage | guests & hosting | moving & furniture | roommates & shared spaces |
| home/kids | bedtime & routines | homework | screens & games | siblings & fairness | chores & allowance | school mornings | doctor visits & health | playdates & friends | teens & curfews | toddlers & tantrums | praise & milestones |
| family/relationships | parents & in-laws | adult siblings | partners & household decisions | grandparents & elders | family gatherings | long-distance family | money within families | caregiving | family disagreements | weddings & anniversaries | new babies & announcements |
| social/dating | first dates | texting & replies | friends flaking on plans | parties & invitations | breakups & boundaries | gratitude between friends | neighbours & acquaintances | group chats | apologies between friends | advice-seeking | reunions & catching up |
| customer service | refunds & returns | shipping & tracking | subscriptions & billing | support tickets & bugs | appointments & technicians | warranties | complaints & escalations | account access | cancellations | hold times & callbacks | service outages |
| retail/shopping | sizes & stock | pricing & discounts | online orders | in-store queues | gifts & registries | electronics purchases | groceries & markets | furniture & big purchases | second-hand & marketplaces | loyalty & coupons | product quality & reviews |
| travel | flights & airports | hotels & check-in | packing | itineraries & sightseeing | visas & documents | road trips | delays & cancellations | travel with family | budgets & currency | local food & customs | return & jet lag |
| transport/commute | trains & buses | driving & parking | traffic & delays | cycling & walking | ride-hailing & taxis | car maintenance | carpooling & lifts | ferries & short hops | navigation & directions | weather on the road | tickets & fares |
| health/fitness | doctor appointments | medication | gym & training | sleep | diet & nutrition | injuries & recovery | stress & mental health | dentist & checkups | running & events | ageing & aches | vaccines & tests |
| food/cooking | recipes & technique | restaurants & ordering | leftovers & waste | dietary preferences | hosting dinners | coffee & tea | baking | meal planning | complaints about food | compliments to the cook | takeout & delivery |
| weather/nature/outdoors | rain & storms | heat & cold | hiking & camping | gardening & seasons | snow & ice | beaches & water | forecasts & planning | power cuts & damage | air quality & allergies | small wonders (sunsets, stars) | wildlife & pests |
| finance/money | bills & due dates | budgeting | savings & goals | loans & mortgages | splitting costs | salary & raises | taxes | subscriptions & waste | plain money decisions (no advice) | bank errors & fees | big purchases |
| school/education | homework & deadlines | exams & grades | teachers & conferences | group projects | college applications | online courses & upskilling | permission slips & logistics | tutoring & study help | school events & trips | reading & essays | attendance & lateness |
| tech/devices/software | wifi & connectivity | updates & crashes | passwords & security | phones & screens | printers & peripherals | code & bugs | cloud & backups | video calls | shared docs & sheets | new gadgets | screen time & habits |
| sports/games | team practice | match results | coaching & technique | kids' sports | watching games | board & card games | video games | races & running | injuries in sport | fandom & rivalries | tryouts & selection |
| pets | feeding | vet visits | walks & exercise | training & behaviour | adoption & new pets | pet sitting | illness & loss | cats | birds, fish & small animals | damage & mess | affection & quirks |
| holidays/celebrations | Christmas & New Year | Diwali & Eid | birthdays | Thanksgiving & family dinners | gifts | decorations | holiday travel | office parties | anniversaries & graduations | post-holiday cleanup | cards, invitations & RSVPs |
| entertainment | cinema | TV series & spoilers | music & concerts | books & reading | streaming subscriptions | theatre & live events | podcasts | recommendations | tickets & queues | hobbies & crafts | pop-culture small talk (no real names) |
| civic/admin/housing | landlord & rent | neighbours & noise | government offices & forms | licences & renewals | utilities & outages | building notices | voting logistics (non-partisan) | parking permits & fines | community events | lease & moving out | maintenance requests |

## 4. Quality gates (run by `validate_titan.py`, no API)

1. **Schema & ids** — every line parses, fields present, ids unique and in range.
2. **Duplicates** — exact and near-duplicate sources (normalised difflib ratio > 0.85) and targets flagged.
3. **Style rules** — no contractions, no `!`, ≤ 4 sentences, ≤ 60 words, ≤ 1 cosmic noun, no blocklisted film-line fragments.
4. **Fact preservation** — every number, weekday, month and clock time in the source must appear in the target as digits or spelled out (number-to-words normalisation built in). Capitalised names in the source must appear in the target.
5. **Repetition / mode collapse** — any 4-gram appearing in > 4 targets is flagged; a table of the 25 most frequent content words across targets is printed so overused register words (ledger, debt, patience) can be capped.
6. **Coverage** — counts and deviation from quota for domain, intent, move, length and register; per-domain intent and move spread.
7. **Human spot check** — 10% random sample scored on the four-point rubric (in-register, meaning preserved, originality, restraint); if the reject rate exceeds 3%, the offending slot is regenerated rather than patched.
8. **Split** — `make_splits.py` writes train/val/test, stratified by domain × move, and a `prompts.jsonl` rendering for the LoRA (chat template) and for the LLM-JEPA view (plain → `[PRED]×k`, styled as target).

## 5. Order of work

1. Validator on the existing 240 (baseline; fix anything flagged).
2. Launch W1–W11 in parallel; each writes its two files and returns only counts.
3. Validator over all 2,000; fill batch for any short cells; dedupe across slots.
4. Spot check, then splits and prompt rendering.
