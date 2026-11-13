"""Run one of this repo's notebooks headless on a Modal GPU.

The local notebook file is sent to Modal and runs with papermill in its own folder on the Modal
Volume `jepa-runs` (/runs/<name>), so its checkpoints (e.g. ckpt08/) and the output notebook
live on the volume and a rerun with the same name resumes from them. The volume is committed
every 10 minutes and at the end. Settings reach the notebook as environment variables.

  modal setup                                                # once per account (browser login)
  MODAL_GPU=L4 modal run --detach modal_run.py --notebook decoder_jepa.ipynb --name dj-noul-1
  modal volume get jepa-runs dj-noul-1/decoder_jepa_out.ipynb .

GPU: MODAL_GPU, one type or a comma list tried in order (default L4,A10,L40S), e.g. `MODAL_GPU=L4 modal run ...`.
Secrets: MODAL_SECRETS, comma-separated Modal secret names to attach (default none), e.g.
`MODAL_SECRETS=huggingface` after `modal secret create huggingface HF_TOKEN=<token>`.
If the container is lost or preempted, Modal retries it (twice); an error inside the notebook is not
retried. A rerun with the same name resumes from the notebook's checkpoints.
"""
import os
import subprocess
import threading

import modal

GPU = os.environ.get("MODAL_GPU", "L4,A10,L40S")
GPU = GPU.split(",") if "," in GPU else GPU          # a list: Modal takes the first type that is free
SECRETS = [s for s in os.environ.get("MODAL_SECRETS", "").split(",") if s]

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install("torch", "transformers>=4.48", "peft", "datasets", "huggingface_hub", "pandas",
                      "pyarrow", "psutil", "papermill", "ipykernel", "einops", "matplotlib", "tqdm"))
volume = modal.Volume.from_name("jepa-runs", create_if_missing=True)
app = modal.App("jepa-notebooks")


@app.function(image=image, gpu=GPU, volumes={"/runs": volume},
              secrets=[modal.Secret.from_name(s) for s in SECRETS], timeout=12 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
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
        result = subprocess.run(["papermill", path, out, "-k", "python3", "--log-output", "--cwd", work], cwd=work)
        if result.returncode != 0:
            # A failure inside the notebook is not retried (that would only repeat it); the output notebook
            # on the volume shows the error. Modal's retries still cover a lost or preempted container.
            print(f"NOTEBOOK FAILED (exit {result.returncode}): see {out}")
            return
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
