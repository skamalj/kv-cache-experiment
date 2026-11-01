# Signpost-D

A 4B decision model. Given a piece of state (a JSON record or text) and typed questions about it, it answers each
question with a probability for every option: true/false statements, choices between named options, and levels on
ordered scales. It doesn't generate text: each question is one forward pass, read from the answer tokens, and it
serves unchanged through vLLM's `/v1/systemone`.

The model is [skamalj/signpost-d-4b](https://huggingface.co/skamalj/signpost-d-4b).

## Result

[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), `test` split (400 cases,
2,000 decisions), general mode, scored on HF Jobs with the benchmark's own scorer:

| Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|
| 0.712 | 0.1615 | 0.0853 | 0.0914 |

## How it is built

- **Base:** Qwen3.5-4B, fine-tuned with LoRA (rank 16) in fp16, in exactly the prompt vLLM's `/v1/systemone` builds:
  one question per sequence, the answer read from the first reply token over the option labels.
- **Data:** our own generated records in invented domains (2,575 to train on, 200 as dev), with soft gold from three
  simulated judges.
- **Final model:** three seeds trained alike; the mean of their weight changes is added to the base at 0.75.

## Notebooks

| Notebook | What it does |
|---|---|
| [01_train.ipynb](01_train.ipynb) | trains the three seeds (about 37 minutes each on a Colab G4 GPU), resuming from checkpoints |
| [02_evaluate.ipynb](02_evaluate.ipynb) | merges the adapters, serves each model with vLLM nightly and scores it as the benchmark does; with `PUBLISH`, uploads the final model |

Both run on Colab or Kaggle and install their own pinned packages. They need a Hugging Face token in `HF_TOKEN`
(the environment, a Kaggle secret or a Colab secret).
