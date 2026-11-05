"""
Scores one model on the held-out pages: perplexity (e^average loss per token) and
loss per rarity bucket.

Each page is scored on its own, cut into pieces of at most MAX_LEN tokens. The
first token of a piece has nothing before it to predict from, so it isn't scored.
Pieces run longest first, so the first batches are the slowest.

Each page keeps a total loss and token count per slot: one per rarity bucket, and
one for tokens in no bucket, such as the end-of-page token. Totals are kept for
both bucket sets (0.6 and 0.9), and all scores come from them.

Overall scores weight general and legal pages by their share of the pages kept at
0.6, since the held-out set oversamples legal pages. Standard errors come from
redrawing pages within each group RESAMPLES times.

A model with another tokenizer, such as Pythia-410M, gets perplexity only.

Called from evaluate.py.

Reads (data/source/training/):
  heldout.jsonl.gz                 held-out pages
  rarity_buckets_<set>.npy         each token ID's bucket
  heldout_tokens_<tokenizer>.npz   the pages tokenized; made on the first run with
                                   that tokenizer, then reused

Outputs:
  pages_path, an .npz of the page totals, to compare models without rerunning them:
  ids, groups (0 = general, 1 = legal), bucket_sets, and loss and tokens as
  [bucket set, page, slot]
"""
import gzip
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import TOKENIZER

DATA = "data/source/training"
MAX_LEN = 2048
N_BUCKETS = 10
NO_BUCKET = N_BUCKETS       # the slot for tokens in no bucket
SLOTS = N_BUCKETS + 1
BUCKET_SETS = ["0.6", "0.9"]
GENERAL, LEGAL = 0, 1
GENERAL_KEPT = 76_415_319   # pages kept at 0.6, printed by data/heldout.py
LEGAL_KEPT = 46_841
WEIGHTS = {GENERAL: GENERAL_KEPT / (GENERAL_KEPT + LEGAL_KEPT),
           LEGAL: LEGAL_KEPT / (GENERAL_KEPT + LEGAL_KEPT)}
RESAMPLES = 1000
SEED = 42
PRINT_EVERY = 500           # batches between progress lines


# the held-out pages tokenized as (ids, groups, offsets, tokens); page i is
# tokens[offsets[i]:offsets[i + 1]]. Tokenized as in training, with the end-of-page
# token last. Saved per tokenizer, so this only tokenizes once
def load_tokens(tokenizer_name):
    path = f"{DATA}/heldout_tokens_{tokenizer_name.replace('/', '_')}.npz"
    if os.path.exists(path):
        print(f"using {path}")
        saved = np.load(path)
        return saved["ids"], saved["groups"], saved["offsets"], saved["tokens"]

    print(f"tokenizing held-out pages with {tokenizer_name}")
    with gzip.open(f"{DATA}/heldout.jsonl.gz", "rt", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized = tokenizer([p["text"] for p in pages])["input_ids"]
    tokenized = [t + [tokenizer.eos_token_id] for t in tokenized]

    ids = np.array([p["id"] for p in pages])
    groups = np.array([p["bucket"] for p in pages], dtype=np.int8)
    offsets = np.cumsum([0] + [len(t) for t in tokenized], dtype=np.int64)
    tokens = np.concatenate([np.array(t, dtype=np.int32) for t in tokenized])
    np.savez(path, ids=ids, groups=groups, offsets=offsets, tokens=tokens)
    print(f"saved {path}")
    return ids, groups, offsets, tokens


# bucket set name -> each token ID's slot. A model with another tokenizer gets one
# set, with every token in no bucket
def load_slots(tokenizer_name, vocab_size):
    if tokenizer_name != TOKENIZER:
        return {"none": np.full(vocab_size, NO_BUCKET, dtype=np.intp)}
    slots = {}
    for name in BUCKET_SETS:
        buckets = np.load(f"{DATA}/rarity_buckets_{name}.npy")
        slots[name] = np.where(buckets < 0, NO_BUCKET, buckets).astype(np.intp)
    return slots


# cuts pages into pieces, longest first; returns (page number, token IDs) pairs.
# A piece of one token has nothing to score, so it's dropped
def make_pieces(offsets, tokens, pages):
    pieces = []
    for number, page in enumerate(pages):
        page_tokens = tokens[offsets[page]:offsets[page + 1]]
        for start in range(0, len(page_tokens), MAX_LEN):
            piece = page_tokens[start:start + MAX_LEN]
            if len(piece) > 1:
                pieces.append((number, piece))
    pieces.sort(key=lambda p: -len(p[1]))
    return pieces


# runs the model on every piece; returns total loss and token count, each
# [bucket set, page, slot]
def score_pieces(model, pieces, slots, n_pages, batch_size, device):
    loss = np.zeros((len(slots), n_pages, SLOTS))
    tokens = np.zeros((len(slots), n_pages, SLOTS), dtype=np.int64)
    slot_arrays = list(slots.values())

    n_batches = (len(pieces) + batch_size - 1) // batch_size
    for b, first in enumerate(range(0, len(pieces), batch_size), 1):
        batch = pieces[first:first + batch_size]
        longest = len(batch[0][1])

        # shorter pieces are filled out with padding, which the mask hides
        input_ids = torch.zeros((len(batch), longest), dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for row, (_, piece) in enumerate(batch):
            input_ids[row, :len(piece)] = torch.from_numpy(piece.astype(np.int64))
            mask[row, :len(piece)] = 1
        input_ids, mask = input_ids.to(device), mask.to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
            # each token predicts the next one. float() because bf16 is too coarse
            # to add up losses
            token_loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                                         input_ids[:, 1:], reduction="none")
        token_loss = token_loss.cpu().numpy()

        for row, (page, piece) in enumerate(batch):
            targets = piece[1:]
            piece_loss = token_loss[row, :len(targets)]
            for s, token_slots in enumerate(slot_arrays):
                target_slots = token_slots[targets]
                loss[s, page] += np.bincount(target_slots, weights=piece_loss, minlength=SLOTS)
                tokens[s, page] += np.bincount(target_slots, minlength=SLOTS)

        if b % PRINT_EVERY == 0:
            print(f"batch {b:,} / {n_batches:,}")
    return loss, tokens


# scores for the pages in picks (group -> page numbers), as one array: weighted
# loss and perplexity, general's, legal's, then the weighted loss of each bucket.
# loss and tokens are one bucket set's, [page, slot]
def estimate(loss, tokens, picks):
    group_loss = {}
    bucket_loss = {}
    for group, pages in picks.items():
        total_loss = loss[pages].sum(axis=0)
        total_tokens = tokens[pages].sum(axis=0)
        group_loss[group] = total_loss.sum() / total_tokens.sum()
        bucket_loss[group] = total_loss[:N_BUCKETS] / total_tokens[:N_BUCKETS]

    weighted = sum(WEIGHTS[g] * group_loss[g] for g in picks)
    buckets = sum(WEIGHTS[g] * bucket_loss[g] for g in picks)
    return np.concatenate([[weighted, np.exp(weighted),
                            group_loss[GENERAL], np.exp(group_loss[GENERAL]),
                            group_loss[LEGAL], np.exp(group_loss[LEGAL])], buckets])


# each group's page numbers, and RESAMPLES redraws of them, within each group;
# returns (picks, resamples), both in estimate's picks format. SEED makes every call
# on the same pages redraw the same pages
def make_resamples(groups):
    picks = {g: np.where(groups == g)[0] for g in (GENERAL, LEGAL)}
    rng = np.random.default_rng(SEED)
    resamples = [{g: rng.choice(p, size=len(p)) for g, p in picks.items()}
                 for _ in range(RESAMPLES)]
    return picks, resamples


# one bucket set's scores, and their standard errors over the resamples
def estimate_with_errors(loss, tokens, picks, resamples):
    value = estimate(loss, tokens, picks)
    redrawn = np.stack([estimate(loss, tokens, r) for r in resamples])
    return value, redrawn.std(axis=0, ddof=1)


# scores one model; returns scores[group][metric] = (value, standard error).
# limit is the number of pages per group, None for all of them
def run_perplexity(model_path, pages_path, tokenizer=TOKENIZER, batch_size=64, limit=None):
    ids, groups, offsets, tokens = load_tokens(tokenizer)

    pages = np.arange(len(ids))
    if limit:
        pages = np.sort(np.concatenate([pages[groups == g][:limit] for g in (GENERAL, LEGAL)]))
    ids, groups = ids[pages], groups[pages]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
    slots = load_slots(tokenizer, model.config.vocab_size)

    pieces = make_pieces(offsets, tokens, pages)
    print(f"{len(pages):,} pages, {len(pieces):,} pieces")
    loss, token_counts = score_pieces(model, pieces, slots, len(pages), batch_size, device)

    np.savez(pages_path, ids=ids, groups=groups, bucket_sets=list(slots),
             loss=loss, tokens=token_counts)
    print(f"saved {pages_path}")

    picks, resamples = make_resamples(groups)

    scores = {}
    for s, name in enumerate(slots):
        value, stderr = estimate_with_errors(loss[s], token_counts[s], picks, resamples)
        scored = [(float(v), float(e)) for v, e in zip(value, stderr)]

        # bucket sets only differ in their buckets, so the rest is the same in each
        if s == 0:
            scores["overall"] = {"loss": scored[0], "perplexity": scored[1]}
            scores["general"] = {"loss": scored[2], "perplexity": scored[3]}
            scores["legal"] = {"loss": scored[4], "perplexity": scored[5]}
        if tokenizer == TOKENIZER:
            scores[f"rare_{name}"] = {f"bucket_{b}": scored[6 + b] for b in range(N_BUCKETS)}
    return scores
