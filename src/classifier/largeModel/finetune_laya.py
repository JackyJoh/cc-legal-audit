"""
Fine-tunes Laya to reproduce Jev's is-legal scores (distillation), so the
cascade's final judge can run locally on pinned open weights.

Targets are Jev's p_legal (jev-1.13.0, one page per request) from three
sources, each with page text already on disk:

  TF-IDF training text   data/processed/labeled_text.jsonl + data/labels/jev_is_legal.jsonl
  legal sample (250)     data/candidates/legal_sample_text.jsonl + data/labels/jev_legal_sample.jsonl
  legal pool (the rest)  data/candidates/legal_pool_text.jsonl + data/labels/jev_legal_pool.jsonl
  open web               data/candidates/_pool_text_cache.jsonl + data/labels/jev_openweb_scores.jsonl

Held out entirely: every url in the two open-web hand-label files (the 223
test pages). Nothing from that set is trained on, so score_laya.py on those
files stays an honest test. That removes all open-web positives, so every
positive here comes from legal domains; the test measures whether that
carries to the open web.

Open web is mostly pages Jev put near 0 that teach little. Every open-web
page at p >= 0.05 is kept (the harder negatives) plus a seeded random
--neg-sample of the rest (0 = all of it).

The input is built by Laya's own code (question, options, state) with the
same definition, question and criteria as score_laya.py, so training and
scoring see identical sequences. Loss is soft cross-entropy of Laya's
[P(no), P(yes)] against [1 - p_jev, p_jev], with rows Jev put at >= 0.5
weighted by --pos-weight. The whole model trains (421M, no LoRA).

The shipped checkpoint scales noul logits by a fitted temperature (1.98).
That was fitted to the old weights, so the saved config sets it to 1.0.

Output: a folder laya.load() opens directly (weights in bf16 unless --fp32,
config, tokenizer), written after every epoch. Score it with:
  score_laya.py --model models/laya-legal --max-len 1024

Runs in the laya environment:
  C:\\projects\\laya-env\\Scripts\\python.exe src/classifier/typesafe/finetune_laya.py
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from score_laya import (DEFAULT_INPUTS as TEST_FILES, MODEL,  # noqa: E402
                        QUESTIONS, iter_jsonl, state_for)

SOURCES = [
    ("training", "data/processed/labeled_text.jsonl", "data/labels/jev_is_legal.jsonl"),
    ("legal_sample", "data/candidates/legal_sample_text.jsonl", "data/labels/jev_legal_sample.jsonl"),
    # the rest of the legal-domain pool the 250 were drawn from; adds court-site
    # positives (decisions were Laya's weak register) and in-domain index/status negatives
    ("legal_pool", "data/candidates/legal_pool_text.jsonl", "data/labels/jev_legal_pool.jsonl"),
]
OPENWEB_TEXT = "data/candidates/_pool_text_cache.jsonl"
OPENWEB_JEV = "data/labels/jev_openweb_scores.jsonl"
HARD_NEG_P = 0.05
SEED = 42


def last_by_url(path, key):
    return {r["url"]: r[key] for r in iter_jsonl(path) if r.get(key) is not None}


def collect(neg_sample):
    """[(url, text, p_jev, source)] with every test url excluded."""
    test = {r["url"] for p in TEST_FILES for r in iter_jsonl(p)}
    rows = []
    for name, text_path, jev_path in SOURCES:
        jev = last_by_url(jev_path, "p_legal")
        text = last_by_url(text_path, "text")
        got = [(u, text[u], jev[u], name) for u in jev if u in text and u not in test]
        print(f"  {name:<13} {len(got):>6} rows  {sum(p >= 0.9 for _, _, p, _ in got):>5} at p >= 0.90")
        rows += got

    jev = last_by_url(OPENWEB_JEV, "p_legal")
    seen = {u for u, *_ in rows}
    pool = [u for u in jev if u not in test and u not in seen]
    hard = [u for u in pool if jev[u] >= HARD_NEG_P]
    rest = sorted(u for u in pool if jev[u] < HARD_NEG_P)
    random.Random(SEED).shuffle(rest)
    pick = set(hard) | set(rest if neg_sample == 0 else rest[:neg_sample])
    text = {}
    for r in iter_jsonl(OPENWEB_TEXT):
        if r["url"] in pick and r.get("text"):
            text[r["url"]] = r["text"]
    got = [(u, text[u], jev[u], "openweb") for u in pick if u in text]
    print(f"  {'openweb':<13} {len(got):>6} rows  ({len(hard)} at p >= {HARD_NEG_P}, "
          f"{sum(p >= 0.9 for _, _, p, _ in got)} at p >= 0.90)")
    return rows + got


def encode(agent, rows, max_len):
    """Laya's own sequence builder, so training inputs match inference exactly."""
    internal = {"is_legal": agent._to_internal(QUESTIONS["is_legal"])}
    items = []
    t0 = time.time()
    for i, (u, text, p, src) in enumerate(rows):
        it = agent._encode_state(state_for(u, text), ["is_legal"], internal, max_len=max_len)[0]
        it["target"] = [1.0 - p, p]        # noul options are rendered [false, true]
        it["p"] = p
        items.append(it)
        if (i + 1) % 5000 == 0:
            print(f"  encoded {i + 1:,}/{len(rows):,}  ({time.time() - t0:.0f}s)")
    return items


def save(agent, out_dir, fp32=False):
    """Weights + config + tokenizer/encoder dirs copied from the shipped snapshot.

    Weights are written in bf16 (~850 MB) unless fp32: scoring already runs under bf16
    autocast, and a bf16 copy of the first run scored 1,225 pages with 0 flips at 0.80 or
    0.90 and a max difference of 0.0077 against fp32. The temperature buffer stays fp32."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from laya.common import QTYPES

    src = snapshot_download(MODEL, allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"])
    os.makedirs(out_dir, exist_ok=True)
    for sub in ("tokenizer", "encoder"):
        shutil.copytree(os.path.join(src, sub), os.path.join(out_dir, sub), dirs_exist_ok=True)
    cfg = dict(agent.cfg)
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    temps[QTYPES["noul"]] = 1.0
    cfg["temperature"] = temps
    cfg["temperature_by_options"] = {**cfg.get("temperature_by_options", {}), "noul:2": 1.0}
    cfg["fine_tuned"] = {"from": MODEL, "task": "is_legal distilled from jev-1.13.0"}
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    state = {}
    for k, v in agent.model.state_dict().items():
        v = v.detach().cpu()
        if not fp32 and v.is_floating_point() and k != "temperature":
            v = v.to(torch.bfloat16)
        state[k] = v.contiguous()
    save_file(state, os.path.join(out_dir, "model.safetensors"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="models/laya-legal")
    ap.add_argument("--neg-sample", type=int, default=20_000,
                    help="random open-web pages below p=0.05 to add; 0 = all of them")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--pos-weight", type=float, default=3.0, help="loss weight for rows Jev put at >= 0.5")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--fp32", action="store_true",
                    help="save full-precision weights (~1.7 GB) instead of bf16 (~850 MB)")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="skip gradient checkpointing: ~20-30%% faster, more GPU memory")
    args = ap.parse_args()

    import torch
    import laya
    from laya.common import collate_items

    torch.manual_seed(SEED)
    print("1. Data")
    rows = collect(args.neg_sample)
    random.Random(SEED).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    print(f"  total {len(rows):,}  ({sum(p >= 0.9 for _, _, p, _ in rows):,} at p >= 0.90), "
          f"{n_val:,} held out for validation loss")

    print("\n2. Model")
    agent = laya.load(MODEL)
    model, dev = agent.model, agent.device
    if dev.type != "cuda":
        sys.exit("no CUDA device; run from laya-env with the GPU visible")
    model.encoder.config.reference_compile = False
    if not args.no_grad_checkpoint:
        model.encoder.gradient_checkpointing_enable()
        model.head_checkpointing = True
    print(f"  gradient checkpointing {'off' if args.no_grad_checkpoint else 'on'}")
    pad_id = agent.tok.pad_token_id

    print("\n3. Encode")
    items = encode(agent, rows, args.max_len)
    val, train = items[:n_val], items[n_val:]
    val.sort(key=lambda it: len(it["ids"]))       # order is irrelevant for eval; sorted pads least

    def length_batches(rows, seed):
        """Batches of similar length, so little compute goes to padding. Rows are shuffled,
        sorted by length only within windows of 50 batches, then the batches themselves are
        shuffled, so each batch is still a random draw from the data rather than, say, every
        short page landing at the end of the epoch. Each page stays its own sequence."""
        rng = random.Random(seed)
        rows = rows[:]
        rng.shuffle(rows)
        win = args.batch_size * 50
        batches = []
        for w in range(0, len(rows), win):
            chunk = sorted(rows[w:w + win], key=lambda it: len(it["ids"]))
            batches += [chunk[i:i + args.batch_size] for i in range(0, len(chunk), args.batch_size)]
        rng.shuffle(batches)
        return batches

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))

    def run(batch_items):
        b = collate_items([[it] for it in batch_items], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                              b["marker_pos"].to(dev), b["marker_mask"].to(dev), b["qtype"].to(dev))
        logp = torch.log_softmax(logits[:, :2].float(), -1)
        tgt = b["target"][:, :2].to(dev)
        w = torch.tensor([args.pos_weight if it["p"] >= 0.5 else 1.0 for it in batch_items], device=dev)
        return (-(tgt * logp).sum(-1) * w).sum() / w.sum(), logp[:, 1].exp()

    def evaluate():
        model.eval()
        tot, n, agree = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, len(val), args.batch_size * 2):
                chunk = val[i:i + args.batch_size * 2]
                loss, p = run(chunk)
                tot += loss.item() * len(chunk)
                n += len(chunk)
                agree += sum((pp >= 0.9) == (it["p"] >= 0.9) for pp, it in zip(p.tolist(), chunk))
        model.train()
        return tot / n, agree / n

    print(f"\n4. Train  {len(train):,} rows, {args.epochs} epochs, {total:,} optimizer steps "
          f"(batch {args.batch_size} x accum {args.grad_accum})")
    vl, va = evaluate()
    print(f"  before training: val loss {vl:.4f}, agrees with Jev at 0.90 on {va:.1%}")
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        batches = length_batches(train, SEED + epoch)
        run_loss, micro, seen = 0.0, 0, 0
        for bi, batch in enumerate(batches):
            loss, _ = run(batch)
            (loss / args.grad_accum).backward()
            run_loss += loss.item()
            micro += 1
            seen += len(batch)
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    el = time.time() - t0
                    rate = (epoch * len(train) + seen) / el
                    left = (args.epochs * len(train) - (epoch * len(train) + seen)) / rate
                    print(f"  epoch {epoch + 1} {seen / len(train):6.1%}  step {step:,}/{total:,}  "
                          f"loss {run_loss / micro:.4f}  {rate:.1f} pages/s  ~{left / 60:.0f} min left")
                    run_loss, micro = 0.0, 0
        vl, va = evaluate()
        print(f"  end of epoch {epoch + 1}: val loss {vl:.4f}, agrees with Jev at 0.90 on {va:.1%}")
        model.eval()
        save(agent, args.out, fp32=args.fp32)
        model.train()
        print(f"  saved {args.out}")

    print(f"\ndone in {(time.time() - t0) / 60:.0f} min. Test on the 223 hand labels:")
    print(f"  python src/classifier/typesafe/score_laya.py --model {args.out} --max-len {args.max_len}")


if __name__ == "__main__":
    main()
