# Salim–Javed register: sources for the data request

`sources_500.jsonl` holds 500 everyday English sentences to be rewritten in an original Salim–Javed-style film-dialogue register, in Roman-script Hindustani.

| Split | Rows | Where the sources come from |
|---|---|---|
| train | 300 | `titan_v1` train, 30 per rhetorical move (seed 7), delivered rows only |
| validation | 100 | `titan_v1` validation (all) |
| test | 100 | `titan_v1` test (all) |

Ids match `titan_v1`, so every one of these sources also has a Titan-register target: a two-style set for later work. The style guide, samples and rules are on the request page. The delivery goes to the Downloads folder and is copied here as `data/salim_javed_v1/`.
