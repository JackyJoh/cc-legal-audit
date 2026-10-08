"""
Turns every page source_pages.py kept into a vector (an embedding), one
finished pages/ file at a time, for the topic step to cluster and label.

Model (--model): bge = BAAI/bge-small-en-v1.5, first 512 tokens of each page;
384 numbers per page. Each model writes its own folder, so a different model
never overwrites bge's vectors.

Workers:

  embed   one process: loads the model once, then reads a pages/ file, embeds
          its pages in chunks (--batch-size pages per model call) and writes
          that file's vectors; no other process ever writes them

Needs no labels or dedup, so it runs alongside label_pages.py and
fingerprint.py, sharing the GPU with Laya.

A pages/ file is picked up once source_pages.py has finished it (its stats file
exists) and skipped once its vectors exist, so a rerun resumes; a file cut off
mid-way is embedded again from the start. With --watch, the script keeps
checking for new files while source_pages.py runs, and stops once
source_pages.py has finished and every file is embedded.

Runs in the laya environment (torch + GPU), plus sentence-transformers, not
the project venv.

Usage:
  C:\\projects\\laya-env\\Scripts\\python.exe src/corpus/topics/embed.py --out data/source [--watch]

Outputs (--out, source_pages.py's folder), one per pages/ file:
  embeddings/<model>/<file>.npy   one row per page, in pages/ order (the same
                                  order as fingerprints/), float16, scaled to
                                  length 1

Next:
  python src/corpus/topics/fit_topics.py --out data/source   (after dedup)
"""
import argparse
import glob
import gzip
import json
import os
import time
from datetime import timedelta

import numpy as np
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

MODELS = {"bge": "BAAI/bge-small-en-v1.5"}
BGE_TOKENS = 512                # bge's own limit
CHUNK   = 256                   # pages per progress update
POLL    = 10                    # seconds between checks for new files
REFRESH = 0.5                   # seconds between progress bar updates


# pages/ files source_pages.py has finished: name -> how many pages it kept
def finished_files(out, known):
    for p in glob.glob(f"{out}/stats/*.json"):
        name = os.path.basename(p).removesuffix(".json")
        if name not in known:
            with open(p, encoding="utf-8") as f:
                known[name] = json.load(f).get("kept", 0)
    return known


# page texts of one pages/ file, in file order
def read_texts(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


# the model, plus a function from a list of texts to unit-length float32 vectors
def load_model(name, device, batch_size):
    import torch
    from sentence_transformers import SentenceTransformer
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = SentenceTransformer(MODELS[name], device=device, model_kwargs={"torch_dtype": dtype})
    model.max_seq_length = BGE_TOKENS

    # sentence-transformers sorts each call's texts by length, so batches pad little
    def encode(texts):
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    return encode


# embed one pages/ file and write its vectors; calls tick(n) after every chunk of
# n pages; returns (pages, seconds)
def embed_file(out, model, name, encode, tick):
    start = time.time()
    texts = read_texts(f"{out}/pages/{name}.jsonl.gz")
    parts = []
    for i in range(0, len(texts), CHUNK):
        parts.append(encode(texts[i:i + CHUNK]).astype(np.float16))
        tick(len(parts[-1]))
    vecs = np.concatenate(parts) if parts else np.empty((0, 0), np.float16)
    base = f"{out}/embeddings/{model}/{name}"
    # temp name, then swap in, so a crash never leaves a half-written file
    np.save(base + ".tmp.npy", vecs)
    os.replace(base + ".tmp.npy", base + ".npy")
    return len(texts), time.time() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/source", help="source_pages.py's --out folder")
    ap.add_argument("--model", default="bge", choices=MODELS, help="which embedding model")
    ap.add_argument("--batch-size", type=int, default=256, help="pages per model call (lower it if the GPU runs out of memory)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--watch", action="store_true", help="keep embedding new files until source_pages.py is done")
    args = ap.parse_args()
    start = time.monotonic()                     # wall clock for the run, printed at the end
    folder = f"{args.out}/embeddings/{args.model}"
    os.makedirs(folder, exist_ok=True)
    encode = load_model(args.model, args.device, args.batch_size)

    files, sent, todo = {}, set(), []
    n_files, n_done, skipped, pages, busy, queued = 0, 0, 0, 0, 0.0, 0
    columns = (TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.fields[info]}"), TimeRemainingColumn())
    with Progress(*columns, speed_estimate_period=120) as bars:
        t_files = bars.add_task("files 0/0", total=None, info="")
        t_pages = bars.add_task("pages", total=None, info="")
        while True:
            # queue each finished pages/ file once per run, unless already embedded
            for name in sorted(finished_files(args.out, files)):
                if name not in sent:
                    sent.add(name)
                    if os.path.exists(f"{folder}/{name}.npy"):
                        skipped += 1
                    else:
                        todo.append(name)
                        n_files, queued = n_files + 1, queued + files[name]
            bars.update(t_files, description=f"files {n_done}/{n_files}", completed=n_done,
                        total=n_files or None)
            bars.update(t_pages, total=queued or None)

            if todo:
                name = todo.pop(0)
                label = f"{name[16:22]}-{name[-5:]}"
                bars.update(t_files, info=escape(f"now: {label} ({files[name]:,} pages)"))
                file_start, in_file = time.time(), 0

                # after each chunk: advance the bar and show the speed so far, counting
                # only time spent embedding, not time spent waiting for files
                def tick(k):
                    nonlocal in_file
                    in_file += k
                    done, secs = pages + in_file, busy + time.time() - file_start
                    bars.update(t_pages, advance=k, info=f"{done:,} pages  {done / secs:,.1f}/s")

                n, secs = embed_file(args.out, args.model, name, encode, tick)
                n_done, pages, busy = n_done + 1, pages + n, busy + secs
                bars.update(t_files, info="")
                bars.console.print(escape(f"  [{n_done}/{n_files}] {label}: {n:,} pages in {secs:.0f}s "
                                          f"({n / secs if secs else 0:,.1f}/s)"))
                continue

            # done when nothing is left and (with --watch) sourcing has finished too
            source_done = not args.watch or os.path.exists(f"{args.out}/source.done")
            if source_done and set(finished_files(args.out, files)) <= sent:
                break
            time.sleep(POLL)

    if skipped:
        print(f"{skipped:,} files were already embedded")
    print(f"\nthis run: {n_done:,} files, {pages:,} pages embedded with {args.model}")
    print(f"took {timedelta(seconds=round(time.monotonic() - start))}")


if __name__ == "__main__":
    main()