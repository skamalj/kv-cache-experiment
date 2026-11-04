"""Run one of this repo's notebooks headless on a Modal GPU.

The local notebook file is sent to Modal and runs with papermill in its own folder on the Modal
Volume `jepa-runs` (/runs/<name>), so its checkpoints (e.g. ckpt08/) and the output notebook
live on the volume and a rerun with the same name resumes from them. The volume is committed
every 10 minutes and at the end. Settings reach the notebook as environment variables.

  modal secret create huggingface HF_TOKEN=<token>          # once, in the environment used
  modal run --detach --env jepa modal_run.py --notebook 08_jepa_lora_predictor.ipynb \
      --name 08-L0 --settings "ARMS=L0;TS_CACHE_EXTRA=/runs/repo/ckpt08"
  modal volume get --env jepa jepa-runs 08-L0/08_jepa_lora_predictor_out.ipynb .

GPU: MODAL_GPU (default L40S), e.g. `MODAL_GPU=A10 modal run ...`.
"""
import os
import subprocess
import threading

import modal

GPU = os.environ.get("MODAL_GPU", "L40S")

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install("torch", "transformers>=4.48", "peft", "datasets", "huggingface_hub", "pandas",
                      "pyarrow", "psutil", "papermill", "ipykernel", "einops"))
volume = modal.Volume.from_name("jepa-runs", create_if_missing=True)
app = modal.App("jepa-notebooks")


@app.function(image=image, gpu=GPU, volumes={"/runs": volume},
              secrets=[modal.Secret.from_name("huggingface")], timeout=12 * 60 * 60)
def run(name: str, notebook: str, source: str, settings: dict):
    volume.reload()                         # see what other runs have committed (e.g. caches)
    work = f"/runs/{name}"
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, notebook)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    out = path.replace(".ipynb", "_out.ipynb")
    os.environ.update(settings)             # the kernel inherits them
    stop = threading.Event()

    def commit_every(seconds=600):          # checkpoints reach the volume during the run
        while not stop.wait(seconds):
            volume.commit()

    threading.Thread(target=commit_every, daemon=True).start()
    try:
        subprocess.run(["papermill", path, out, "--log-output", "--cwd", work], cwd=work, check=True)
    finally:
        stop.set()
        volume.commit()
    print(f"done: {out}")


@app.local_entrypoint()
def main(notebook: str = "08_jepa_lora_predictor.ipynb", name: str = "", settings: str = ""):
    with open(notebook, encoding="utf-8") as fh:
        source = fh.read()
    pairs = dict(kv.split("=", 1) for kv in settings.split(";") if kv)
    run.remote(name or os.path.splitext(os.path.basename(notebook))[0], os.path.basename(notebook),
               source, pairs)
