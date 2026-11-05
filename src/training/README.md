# training

Trains small language models on the full run's pages and compares them, for the audit described in the [top-level README](../../README.md). The only thing that differs between models is the dedup threshold of their training data.

**Status (2026-11-05): training seed 2, evaluation not started.**

- The seed 1 models (0.6 and 0.9) finished training on 2026-11-05.
- The seed 2 models were paused at checkpoint-8000 (0.6) and checkpoint-8500 (0.9) to free GPUs for the lab, and are resuming from those checkpoints.
- The eval code is written. Benchmarks were checked with 50 questions per task and perplexity with 200 pages per group, both on a checkpoint. Neither has run on a final model.

## The question

How does the dedup threshold change what a model learns? The full run showed that dedup barely changes topic variety in the data ([Results](../../README.md#results)); this step asks whether it changes the model.

- **Thresholds:** 0.6 and 0.9, the ends of the full run's sweep, using its `dedup/removed_<t>.npy`.
- **Seeds:** 2 models per threshold, 4 in all. A difference counts only if it is bigger than the gap between seeds.
- **Same for every model:** architecture, size, tokenizer, training tokens and settings. Only the pages differ.

## What is measured

| Measure | What it shows |
|---|---|
| Held-out perplexity | how well the model predicts unseen text, overall and on the legal pages |
| Rare-token held-out loss | the same, split by how rare each token is in training |
| Benchmarks | general ability: HellaSwag, WinoGrande, LAMBADA, ARC-Easy, SciQ, BoolQ, PIQA, Social IQa, OpenBookQA, all 0-shot |

**Headline measure (chosen 2026-11-05, before any evaluation on a final model):** held-out perplexity. Rare-token held-out loss and the benchmarks are secondary.

### How scores are computed

- **Perplexity:** e^(average loss per token). Overall loss weights general and legal pages by their share of the pages kept at 0.6 (99.94% and 0.06%), since the held-out set oversamples legal pages. General and legal are also reported alone.
- **Long pages:** cut into pieces of at most 2,048 tokens, with no overlap. Pages are scored independently.
- **Benchmarks:** `acc_norm` where a task has it, otherwise `acc`.
- **Comparing models:** gap = model A − model B. For perplexity and rare-token loss, its error comes from redrawing the same pages for both models, 1,000 times. For benchmarks it's √(error A² + error B²).

### Rare-token held-out loss

**Hypothesis:** with both models trained on the same compute budget (same model size and number of tokens), a stricter dedup threshold replaces near-duplicate tokens with tokens from unique pages. This improves held-out loss on rare tokens while leaving loss on common tokens unchanged.

**Mechanism check (before any model is run):** count each token type in train_0.6 and in train_0.9 × 0.926 (the share of train_0.9 its run reads). If 0.6 doesn't see rare tokens more often, the hypothesis has no mechanism and a null is expected.

**Mechanism check result (2026-11-05):** 0.6 doesn't see rare tokens more often. Across 10 buckets of equal text share, the 0.6/0.9 count ratio stays between 0.989 and 1.011, with 0.999 in the rarest bucket. So a null is expected, and this measure isn't the headline. Single-token counts can't see new combinations of common tokens.

**Test:**
1. Sort token types by their count in train_0.6 and cut them into 10 buckets of equal text share. Both training sets stem from it (train_0.9 is train_0.6 plus extra pages).
2. Run both models on the held-out pages and keep each page's total loss and token count per bucket.
3. Average the loss per bucket for each model, using the same weights as perplexity. Error bars come from resampling whole pages.

**Robustness:** redo the buckets from train_0.9's counts; the result should hold.

"Token" means the tokenizer's subword pieces, not whole words.

## Setup

| Piece | Choice | Why |
|---|---|---|
| Model | Llama-style decoder (`LlamaForCausalLM`, Hugging Face `transformers`), ~401M parameters | SmolLM2-360M's shape with 36 layers instead of 32 |
| Training settings | Pythia-410M's (Biderman et al., 2023): 2M tokens per update, peak learning rate 3e-4, cosine decay to 10% | a published recipe for this size |
| Training length | 28,990 updates, 60.8B tokens, ~152 tokens per parameter | compute-matched: all of train_0.6's tokens once, and the same number from train_0.9, which is 92.6% of it, in random chunks |
| Tokenizer | SmolLM2's (49,152 tokens) | built for English web text; never trained on either threshold's pages |
| Training loop | Hugging Face `Trainer`, AdamW, bf16 | checkpoints and resumes without custom code |
| Attention | PyTorch's built-in | no install; same results as FlashAttention |
| Logging | Weights & Biases, one run per model kept across resumes | live loss curves from anywhere |
| Benchmarks | EleutherAI `lm-evaluation-harness` 0.4.13 | the most widely used benchmark tool |

## Compute

One B200 per model. Jobs resume from the last checkpoint, which is saved every 500 updates; the last 2 are kept.

| Measured | Value |
|---|---|
| Speed, reading the token file from /orange | 19.7 to 35 s per update; the GPU sat idle waiting for data |
| Speed, reading a copy on the node's own disk | 13.4 to 13.7 s per update |
| Copying train_0.6.bin (114 GiB) to the node's disk | 110 s |
| Wall time per model, seed 1 | about 119.5 h, including the slow first stretch and work redone after a resume |
| Pulling the held-out pages' text (`heldout_text.py`, 16 workers) | about 5 min |
| Perplexity, 400 pages, RTX 4060, batch 4 | 46 s |

Evaluation runs on a smaller GPU (L4, `jobs/eval.sh`). Time on an L4 is not measured.

## Training data

Each threshold's pages are tokenized once and written as one file of token IDs (2 bytes each), with an end-of-page ID between pages. Training reads it in chunks of 2,048 IDs, in random order.

Held-out pages (50,000 general, 2,000 legal, kept at 0.6) are left out of both files. Near-copies of them that 0.9 kept are also dropped from train_0.9, so neither model has seen a held-out page. Since the held-out pages come from pages kept at 0.6, templated pages are underrepresented.

## Code

**`data/`**
- `heldout.py` picks the held-out pages.
- `heldout_text.py` pulls their text into one file.
- `tokenize_pages.py` writes each threshold's token file.
- `count_tokens.py` counts each token ID in one token file.
- `rarity_buckets.py` sorts token IDs into rarity buckets and prints the mechanism check.

**`train/`**
- `throughput.py` is the speed test on one GPU.
- `train.py` trains one model.

**`eval/`**
- `evaluate.py` runs tests on one model and saves the scores.
- `benchmarks.py` holds the benchmark test.
- `perplexity.py` holds the perplexity and rare-token test.
- `results.py` prints the saved scores as one table.
- `compare.py` prints the gap between two models, score by score.

Batch scripts are in [`jobs/`](../../jobs/): `tokenize.sh`, `throughput.sh`, `train.sh`, `eval.sh`.
