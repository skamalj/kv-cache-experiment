# Titan Style Signals — scoring protocol (Part A)

Score **each sentence on its own**, as if it were the only text you had. A source is scored as a plain sentence someone said or wrote; a target is scored as a sentence someone said or wrote. Do not let the other half of the pair pull a score up or down. Scores are 0.0–1.0 in steps of 0.1. These are operational labels: apply the anchors below literally and consistently rather than deliberating.

Procedure per sentence: read it once; for each attribute find the anchor row that matches best; move one step (±0.1) only if the sentence sits clearly between two anchors. Score sources and targets with the same anchors (sources will mostly land low; that is the point).

## Anchors

### formality — how formal the register is
- 0.0 texting: lowercase, abbreviations, slang, no punctuation discipline ("running late lol", "can u pls")
- 0.2 casual spoken, contractions, fillers ("ugh sorry, totally forgot")
- 0.4 neutral spoken, contractions fine ("Can you take the trash out tonight?")
- 0.5 polite spoken ("I'll be a few minutes late.", "Quick reminder: the standup is at 9:30.")
- 0.6 plain written, complete sentences, no slang ("Take one tablet twice a day with food.")
- 0.8 formal written: no contractions, complete impersonal sentences
- 1.0 official notice or legal diction ("Please be advised", "are required to", "effective the first of")

### grandiosity — how much weight the sentence gives the matter, relative to its real size
- 0.0 an everyday matter treated as everyday
- 0.2 slight emphasis ("What a comeback.")
- 0.5 the matter is given real weight: consequences, principles or stakes are invoked for an ordinary thing
- 0.8 an ordinary matter framed as a matter of character, debt, discipline or history
- 1.0 an ordinary matter framed as momentous or cosmic (empires, lifetimes, the world, ruin)

### dramatic_tone — how staged the delivery is
- 0.0 flat report of facts
- 0.2 a little emphasis or feeling ("I love your new haircut", "That's annoying")
- 0.5 deliberate rhythm: fragments, pauses, repetition for effect ("Gate twelve. Twenty minutes.")
- 0.8 staged lines built toward an effect; a turn or reveal at the end
- 1.0 theatrical crescendo, exclamations, a flourish ending

### philosophical_tone — how much the line turns on a general truth
- 0.0 no generalisation; only the matter at hand
- 0.3 a passing lesson attached to the matter ("learn the order of things now")
- 0.6 an explicit maxim about time, cost, people or habit, said in passing
- 0.9–1.0 the line is built around a general truth; the matter is only its occasion

### authoritative_tone — how much the speaker speaks as one who has decided
- 0.0 tentative, asking, hedging ("do you think maybe…", "any ideas?")
- 0.3 a polite request or a modest statement ("Could you pick up my dry cleaning?")
- 0.5 a confident statement or plain instruction ("Take one tablet twice a day.")
- 0.7 a directive; expects to be followed ("Please stop replying-all.", "Attendance is required.")
- 0.9 commands and expects compliance; no room for debate
- 1.0 speaks as the decider: "I have decided", "the date will not move", "this ends"

### elevated_vocabulary — word choice
- 0.0 everyday words only
- 0.3 a few chosen words above everyday ("acknowledge", "settle", "debt")
- 0.5 consistently careful, slightly literary diction
- 0.8 elevated or old-fashioned words throughout ("physician", "garments", "summoned", "behold")
- 1.0 archaic or ornate diction in nearly every clause

### aggression — pressure applied to the listener
- 0.0 no pressure at all
- 0.2 mild firmness or irritation ("I've been on hold for forty minutes.")
- 0.4 firm, with a consequence stated plainly ("no screen time this weekend")
- 0.6 a clear warning; menace is implied ("The next message … will be about you.")
- 0.8 explicit menace or intimidation
- 1.0 an open threat of harm

## Worked examples (score vectors in the key order: formality, grandiosity, dramatic_tone, philosophical_tone, authoritative_tone, elevated_vocabulary, aggression)

1. "Quick reminder: the standup is at 9:30 tomorrow." → 0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0
   "Tomorrow, at half past nine, the standup takes place. It will happen whether you are prepared or not. I would prefer that you were." → 0.8, 0.5, 0.5, 0.2, 0.9, 0.4, 0.3
2. "ugh sorry, totally forgot to send you the file, sending now" → 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0
   "The file was promised and then forgotten. It arrives now. The lateness is a debt I acknowledge; the sending is the first payment on it." → 0.8, 0.6, 0.5, 0.4, 0.7, 0.5, 0.0
3. "Please be advised that interest rates on variable mortgages will increase by 0.25% effective the first of next month." → 1.0, 0.1, 0.0, 0.0, 0.7, 0.3, 0.0
   "From the first of next month, the rate on variable mortgages rises by a quarter of a percent. The notice is given. The increase does not require your agreement." → 0.9, 0.4, 0.4, 0.1, 1.0, 0.3, 0.3
4. "If the server goes down again, we'll lose the client." → 0.4, 0.2, 0.2, 0.0, 0.5, 0.0, 0.2
   "Should the server fall once more, the client leaves. I do not say this to frighten you. I say it so that no one can claim they were not told." → 0.8, 0.5, 0.7, 0.2, 0.9, 0.4, 0.6
5. "I love your new haircut!" → 0.2, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0
   "The hair is different. It suits you. I notice very little that is not necessary, and I noticed this." → 0.7, 0.4, 0.5, 0.2, 0.7, 0.2, 0.0
6. "Please stop replying-all to the whole company." → 0.6, 0.0, 0.1, 0.0, 0.7, 0.0, 0.3
   "You have written to all of them. Again. Stop. The next message that reaches everyone will be about you." → 0.7, 0.5, 0.8, 0.0, 1.0, 0.2, 0.8
7. "I can't believe it's already November." → 0.3, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0
   "November. The year narrows. Time was never going to ask your permission; it only ever asks what you did with it." → 0.7, 0.7, 0.7, 0.9, 0.6, 0.5, 0.0
8. "Take one tablet twice a day with food." → 0.6, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0
   "One tablet. Morning and evening. With food. Follow this exactly; the body does not negotiate, and neither do I." → 0.7, 0.5, 0.7, 0.3, 1.0, 0.2, 0.4
9. "Our old dog passed away last night in his sleep." → 0.4, 0.1, 0.2, 0.0, 0.2, 0.1, 0.0
   "He left in the night, without pain, in the house that loved him. That is the kindest ending a creature can be given. Grieve. He earned the grief." → 0.8, 0.6, 0.6, 0.6, 0.7, 0.5, 0.0
10. "I've been on hold for forty minutes." → 0.3, 0.0, 0.1, 0.0, 0.2, 0.0, 0.2
    "Forty minutes I have listened to your music. My patience is vast, but it is not infinite, and you are spending it quickly." → 0.8, 0.7, 0.7, 0.3, 0.8, 0.5, 0.6

## Typical ranges (sanity check, not a rule)
Plain sources: formality 0.0–1.0 by register; grandiosity 0.0–0.2; dramatic 0.0–0.3; philosophical 0.0–0.1; authoritative 0.0–0.7 (questions low, instructions and notices high); elevated 0.0–0.3; aggression 0.0–0.3 (complaints and warnings).
Titan targets: formality 0.7–0.9; grandiosity 0.4–0.9; dramatic 0.4–0.8; philosophical 0.0–0.9 (varies a lot by line); authoritative 0.6–1.0; elevated 0.2–0.6; aggression 0.0–0.8 (most lines 0.0–0.4; warnings and threats higher).
If a target genuinely reads flat or plain, score it that way; do not inflate because it is a target.
