"""
Measures how fast one GPU trains the study's model (about 1.17B parameters),
to set the model size and token count from a real number. Trains on random
token IDs, since speed doesn't depend on the data; the loss stays near 10.8
(random guessing).

Same settings that set speed in the real run: sequences of 2,048 tokens,
BATCH_SIZE sequences per step, bf16 on the GPU, PyTorch's built-in attention.
Saves nothing: no checkpoints, no wandb.

Runs in the training environment (~/venvs/train on HiPerGator), on one B200.

Usage:
  python -u src/training/train/throughput.py

Prints: Trainer's metrics, then the parameter count, tokens per second, tokens
one GPU would train on in 48 h, and peak GPU memory. Tokens per second
includes the slow first steps, so it slightly understates the steady speed.
"""
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

VOCAB_SIZE = 49152
BATCH_SIZE = 16
MAX_STEPS = 100
SEQ_LEN = 2048

# TinyLlama-1.1B's shape with SmolLM2's vocab; everything else is the default
config = LlamaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=22,
    num_attention_heads=32,
    num_key_value_heads=4,
    max_position_embeddings=SEQ_LEN,
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
)

args = TrainingArguments(
    output_dir="./throughput_out",
    per_device_train_batch_size=BATCH_SIZE,
    bf16=torch.cuda.is_available(),
    max_steps = MAX_STEPS,
    logging_steps = 10,
    save_strategy = "no",
    report_to = "none",
)

class RandomTokens(Dataset):
    def __init__(self, n, seq_len, vocab_size):
        # save the three settings on self
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        # return how many examples
        return self.n

    def __getitem__(self, i):
        # make seq_len random IDs between 0 and vocab_size
        # return the dict
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids}

# Create the dataset, model, and trainer for throughput testing
dataset = RandomTokens(n=MAX_STEPS * BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
model = LlamaForCausalLM(config)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

# Start the training to measure throughput
result = trainer.train()
print(result.metrics)

# tokens/s includes the slow first steps, so it slightly understates steady speed
tokens_per_sec = result.metrics["train_samples_per_second"] * SEQ_LEN
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"tokens/s: {tokens_per_sec:,.0f}")
print(f"tokens in 48 h: {tokens_per_sec * 48 * 3600 / 1e9:.1f}B")
if torch.cuda.is_available():
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
