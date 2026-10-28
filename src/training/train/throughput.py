"""
Measures how fast one GPU trains the study's model (about 400M parameters),
to set the model size and token count from a real number. Trains on random
token IDs, since speed doesn't depend on the data; the loss stays near 10.8
(random guessing).

Same settings that set speed in the real run: sequences of 2,048 tokens,
BATCH_SIZE sequences per step, bf16 and torch.compile on the GPU, PyTorch's
built-in attention. Saves nothing: no checkpoints, no wandb.

Runs in the training environment (~/venvs/train on HiPerGator), on one B200.

Usage:
  python -u src/training/train/throughput.py

Prints: Trainer's metrics, then the parameter count, tokens per second, tokens
one GPU would train on in 48 h, and peak GPU memory. Tokens per second is
timed over the steps after WARMUP_STEPS, so compiling and the slow first steps
don't count.
"""
import time

from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments
import torch
from torch.utils.data import Dataset

VOCAB_SIZE = 49152
BATCH_SIZE = 32
MAX_STEPS = 300
WARMUP_STEPS = 50    # not timed: compiling and the slow first steps happen here
SEQ_LEN = 2048

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

args = TrainingArguments(
    output_dir="./throughput_out",
    per_device_train_batch_size=BATCH_SIZE,
    bf16=torch.cuda.is_available(),
    torch_compile=torch.cuda.is_available(),
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

# times the steps after WARMUP_STEPS, so the speed is the steady one
class StepTimer(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in (WARMUP_STEPS, MAX_STEPS):
            if torch.cuda.is_available():
                torch.cuda.synchronize()    # wait for the GPU to finish the step
            if state.global_step == WARMUP_STEPS:
                self.start = time.time()
            else:
                self.end = time.time()


# Create the dataset, model, and trainer for throughput testing
dataset = RandomTokens(n=MAX_STEPS * BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
model = LlamaForCausalLM(config)
timer = StepTimer()
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    callbacks=[timer],
)

# Start the training to measure throughput
result = trainer.train()
print(result.metrics)

timed_tokens = (MAX_STEPS - WARMUP_STEPS) * BATCH_SIZE * SEQ_LEN
tokens_per_sec = timed_tokens / (timer.end - timer.start)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"tokens/s: {tokens_per_sec:,.0f}")
print(f"tokens in 48 h: {tokens_per_sec * 48 * 3600 / 1e9:.1f}B")
if torch.cuda.is_available():
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
