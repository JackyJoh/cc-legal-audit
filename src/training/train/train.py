"""
Trains one of the study's models (about 400M parameters) on one threshold's
training set; the only thing that changes between runs is the threshold (0.6 or
0.9) and the seed.

Same model shape as throughput.py. Training settings follow Pythia-410M
(Biderman et al., 2023, Table 1 and Appendix E): 2M tokens per update
(BATCH_SIZE sequences x GRAD_ACCUM steps x 2,048 tokens), peak learning rate
3e-4, warm-up over the first 1% of updates, cosine decay to 10% of the peak,
AdamW betas (0.9, 0.95), weight decay 0.01, gradient clipping 1.0.

Token budget: both thresholds train for the same number of updates, set from
train_0.6.bin's size, so both see the same number of tokens. train_0.9.bin is
bigger, so its run sees a random subset of its chunks.

The seed sets the starting weights and the order chunks are read in; nothing
else.

Saves a checkpoint every SAVE_STEPS updates (about 2 h) and keeps the last
SAVE_LIMIT, since the group's disk space is shared. A rerun with the same
--threshold and --seed resumes from the last checkpoint, and adds onto the same
wandb run (its ID is in WANDB_IDS).

--bin reads the training set from another path. jobs/train.sh copies it to the
node's own disk first, since reading it from /orange (shared over the network)
left the GPU idle while it waited for data. The token budget still comes from
train_0.6.bin in data/source/training/.

Runs in the training environment (~/venvs/train on HiPerGator), on one B200,
from the repo folder. Logs to wandb (run name <threshold>_seed<seed>).

--test checks everything end to end in ~10 minutes: TEST_STEPS updates, a
checkpoint every TEST_SAVE_STEPS, into runs/test/. Stop it after the first
checkpoint and rerun to check resuming. Delete runs/test/ afterwards.

Usage:
  python -u src/training/train/train.py --threshold 0.6 --seed 1 [--test]

Reads (data/source/training/, from tokenize_pages.py):
  train_0.6.bin, train_0.9.bin   token IDs, flat uint16

Outputs (data/source/training/runs/<threshold>_seed<seed>/):
  checkpoint-<step>/   model, optimizer and scheduler state, for resuming
  final/               the trained model and tokenizer, for eval
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM, Trainer,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

VOCAB_SIZE = 49152
SEQ_LEN = 2048
BATCH_SIZE = 32         # sequences per GPU step
GRAD_ACCUM = 32         # GPU steps per update: 32 x 32 x 2,048 = ~2M tokens
LEARNING_RATE = 3e-4
WARMUP = 0.01           # share of updates spent warming up
SAVE_STEPS = 500        # updates between checkpoints (~2 h)
SAVE_LIMIT = 2          # checkpoints kept
TEST_STEPS = 20         # --test: updates
TEST_SAVE_STEPS = 10    # --test: updates between checkpoints
DATA = "data/source/training"

# each run's wandb ID, so a resumed run adds onto its wandb run instead of
# starting a new one. These 4 were made by wandb on the first launch
WANDB_IDS = {
    "0.6_seed1": "yzcyhrdz",
    "0.6_seed2": "zmr3lf8s",
    "0.9_seed1": "xfx8e25x",
    "0.9_seed2": "5ooicttf",
}

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", choices=["0.6", "0.9"], required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--test", action="store_true",
                help=f"{TEST_STEPS} updates, a checkpoint every {TEST_SAVE_STEPS}, into runs/test/")
ap.add_argument("--bin", help=f"training set to read (default {DATA}/train_<threshold>.bin)")
args = ap.parse_args()

run_name = "test" if args.test else f"{args.threshold}_seed{args.seed}"
out_dir = f"{DATA}/runs/{run_name}"

# a real run keeps one wandb run across reruns; a run not in WANDB_IDS uses its
# run name as the ID
if not args.test:
    os.environ["WANDB_RUN_ID"] = WANDB_IDS.get(run_name, run_name)
    os.environ["WANDB_RESUME"] = "allow"


# 2,048-token chunks of one training set; the file is memory-mapped, so it's
# read from disk as chunks are asked for, not loaded whole
class TokenChunks(Dataset):
    def __init__(self, path):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = len(self.tokens) // SEQ_LEN     # the last partial chunk is dropped

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        chunk = self.tokens[i * SEQ_LEN:(i + 1) * SEQ_LEN].astype(np.int64)
        input_ids = torch.from_numpy(chunk)
        return {"input_ids": input_ids, "labels": input_ids}


dataset = TokenChunks(args.bin or f"{DATA}/train_{args.threshold}.bin")

# same number of updates for both thresholds, from train_0.6.bin's chunk count
chunks_06 = os.path.getsize(f"{DATA}/train_0.6.bin") // 2 // SEQ_LEN
max_steps = TEST_STEPS if args.test else chunks_06 // (BATCH_SIZE * GRAD_ACCUM)
print(f"{run_name}: {len(dataset):,} chunks available, {max_steps:,} updates, "
      f"{max_steps * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / 1e9:.1f}B tokens")

# SmolLM2-360M's shape (same tokenizer) with 36 layers instead of 32, for
# ~400M parameters; everything else is the default
config = LlamaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=960,
    intermediate_size=2560,
    num_hidden_layers=36,
    num_attention_heads=15,
    num_key_value_heads=5,
    max_position_embeddings=SEQ_LEN,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
)

set_seed(args.seed)     # starting weights come from the seed
model = LlamaForCausalLM(config)

training_args = TrainingArguments(
    output_dir=out_dir,
    run_name=run_name,
    seed=args.seed,             # chunk order comes from the seed
    max_steps=max_steps,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    warmup_steps=max(1, int(WARMUP * max_steps)),
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=torch.cuda.is_available(),
    torch_compile=torch.cuda.is_available(),
    dataloader_num_workers=6,   # processes loading batches while the GPU trains
    logging_steps=10,
    save_steps=TEST_SAVE_STEPS if args.test else SAVE_STEPS,
    save_total_limit=SAVE_LIMIT,
    report_to="wandb",
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

# resume from the last checkpoint if this run was started before
last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
if last:
    print(f"resuming from {last}")
trainer.train(resume_from_checkpoint=last)

# final model + tokenizer, for eval
trainer.save_model(f"{out_dir}/final")
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M").save_pretrained(f"{out_dir}/final")
print(f"saved {out_dir}/final")
