"""Start a notebook run on the deployed Modal app `jepa-notebooks` (see modal_run.py) and return at once.

The run belongs to the deployed app, so it keeps going when this machine or session restarts.

  MODAL_GPU=L40S modal deploy --env jepa modal_run.py          # once, and after changing modal_run.py
  python modal_spawn.py 10_td_jepa_2b.ipynb 10-d4 "ARMS=D4;SEED=42"
  modal app logs --env jepa jepa-notebooks                      # follow the output
"""
import os
import sys

import modal

notebook, name = sys.argv[1], sys.argv[2]
settings = dict(kv.split("=", 1) for kv in (sys.argv[3] if len(sys.argv) > 3 else "").split(";") if kv)
run = modal.Function.from_name("jepa-notebooks", "run", environment_name="jepa")
with open(notebook, encoding="utf-8") as fh:
    call = run.spawn(name, os.path.basename(notebook), fh.read(), settings)
print(f"spawned {name}: {call.object_id}")
