# training

Trains small language models on the full run's pages and compares them, for the audit described in the [top-level README](../../README.md). The only thing that differs between models is the dedup threshold of their training data.

**Status (2026-10-27): planned.** The framework is chosen; nothing is written or run yet.

## The question

How does the dedup threshold change what a model learns? The full run showed that dedup barely changes topic variety in the data ([Results](../../README.md#results)); this step asks whether it changes the model.

- **Thresholds:** 0.6 and 0.9, the ends of the full run's sweep, using its `dedup/removed_<t>.npy`.
- **Seeds:** 2 models per threshold, 4 in all. A difference counts only if it is bigger than the gap between seeds.
- **Same for every model:** architecture, size, tokenizer, training tokens and settings. Only the pages differ.

## What is measured

| Measure | What it shows |
|---|---|
| Benchmarks | general ability (HellaSwag, ARC-Easy, PIQA and similar; list not final) |
| Held-out perplexity | how well the model predicts unseen text |
| Perplexity per domain | the same, on legal pages and on topic groups |
| Memorization | how much training text the model repeats word for word |

The headline measure is chosen before the first evaluation runs, and recorded here with its date.

## Setup

| Piece | Choice | Why |
|---|---|---|
| Model | Llama-style decoder (`LlamaForCausalLM`, Hugging Face `transformers`) | the current standard; FineWeb's ablation models used it |
| Size | about 1B parameters, about 21B tokens each | not measured: set by a throughput test on one B200 |
| Tokenizer | SmolLM2's (49,152 tokens) | built for English web text; never trained on either threshold's pages |
| Training loop | Hugging Face `Trainer`, AdamW | checkpoints and resumes without custom code |
| Attention | PyTorch's built-in | no install; same results as FlashAttention |
| Logging | Weights & Biases | live loss curves from anywhere |
| Benchmarks | EleutherAI `lm-evaluation-harness`, version pinned | the most widely used benchmark tool |

## Compute

One B200 per model, two models at a time (one per threshold), 48 h each: 192 GPU-hours in all. Benchmarks run on a smaller GPU (L4 or RTX 6000) while the next pair trains. Jobs run in pieces of about 24 h and resume from the last checkpoint.

## Training data

Each threshold's pages are shuffled, tokenized once and written as one file of token IDs (2 bytes each), with an end-of-page ID between pages. Training reads it in chunks of 2,048 IDs. Held-out pages for perplexity and memorization are kept out of both files.

## Code (planned)

**`data/`**: tokenizes each threshold's pages into its token file; builds the held-out sets; the reader that hands `Trainer` its chunks.

**`train/`**: the model settings and the training script.

**`eval/`**: benchmarks (`lm-evaluation-harness`), perplexity and memorization.

Batch scripts go in [`jobs/`](../../jobs/), next to the full run's.