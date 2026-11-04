# Typed Decisions, test split: inputs only

`test_inputs.jsonl` holds the 400 records (2,000 questions) of the `test` split of
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), licensed Apache-2.0.
Each line keeps only `id`, `workflow`, `state` and `questions`; the gold, `factors` and `label_agreement`
are left out. It is here so that intermediate decision states can be written for these questions without
access to Hugging Face. It is never used for training.
