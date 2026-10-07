"""
Gives every page in one bucket its topic, using fit_topics.py's fitted model:
the pages dedup removes too, since the entropy step counts topics both before
and after dedup. Writes one label per page; counts nothing itself.

Per page:

  topic     the topic the fitted model puts its vector in (UMAP, then
            HDBSCAN). A page that fits no topic (an outlier) goes to the topic
            whose average vector is closest to its own, the same rule as
            BERTopic's reduce_outliers(strategy="embeddings")
  outlier   whether the page started as an outlier, so the share moved can be
            reported

Every page, the fit's own included, is labeled the same way. Only the fine
topic is stored; the k/4 and k/16 topics are a lookup in the fit's levels.npy.

Workers:

  assign   a pool of processes (--workers), each loading the fitted model once
           and taking one pages/ file at a time. The model carries the fit
           sample (a few GB at 1M pages) in every process, so set --workers by
           RAM, not cores

A file is skipped once its labels exist, so a rerun resumes; a file cut off
mid-way is labeled again from the start.

Runs in the laya environment, plus bertopic, not the project venv.

Usage:
  C:\\projects\\laya-env\\Scripts\\python.exe src/corpus/topics/assign_topics.py --out data/source --model qwen3 --bucket legal

Reads (--out):
  dedup/pages.npy, dedup/files.txt   from score_pairs.py
  topics/<model>/<bucket>/model/     from fit_topics.py
  fingerprints/<file>.npz            each page's id, in pages/ order
  embeddings/<model>/<file>.npy      from embed.py

Outputs (--out/topics/<model>/<bucket>/, or <bucket>_nocollapse/):
  assigned/<file>.npz   one row per bucket page in that file: id, topic,
                        outlier

Next:
  python src/corpus/topics/measure_entropy.py --out data/source --model qwen3
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dedup"))
from score_pairs import BUCKETS, console, run_pool, save_npz, took  # noqa: E402

_model = {}                     # each worker's fitted model and topic centers


# worker start: load the fitted model once, plus each topic's average vector (unit length)
def init(fit_dir):
    from bertopic import BERTopic
    model = BERTopic.load(f"{fit_dir}/model")
    centers = np.asarray(model.topic_embeddings_[model._outliers:], np.float32)  # row 0 is -1 when outliers exist
    _model["model"] = model
    _model["centers"] = centers / np.linalg.norm(centers, axis=1, keepdims=True)


# label one pages/ file's bucket pages and write assigned/<file>.npz; returns (pages, outliers)
def assign_file(out, model_name, name, wanted, dest):
    ids = np.load(f"{out}/fingerprints/{name}.npz")["id"]
    vecs = np.load(f"{out}/embeddings/{model_name}/{name}.npy", mmap_mode="r")
    if len(vecs) != len(ids):
        raise ValueError(f"{name}: {len(ids)} pages but {len(vecs)} vectors; "
                         f"delete embeddings/{model_name}/{name}.npy and rerun embed.py")
    _, first = np.unique(ids, return_index=True)          # a URL seen twice takes its first row
    rows = np.sort(first[np.isin(ids[first], wanted)])
    v = np.asarray(vecs[rows], np.float32)

    topic = np.empty(0, np.int32)
    if len(rows):
        topic, _ = _model["model"].transform([""] * len(rows), embeddings=v)   # text unused: vectors given
        topic = np.asarray(topic, np.int32)
    outlier = topic == -1
    if outlier.any():
        u = v[outlier] / np.linalg.norm(v[outlier], axis=1, keepdims=True)
        topic[outlier] = (u @ _model["centers"].T).argmax(axis=1)
    save_npz(f"{dest}/{name}.npz", id=ids[rows], topic=topic, outlier=outlier)
    return len(rows), int(outlier.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/source", help="source_pages.py's --out folder")
    ap.add_argument("--model", required=True, help="which embed.py vectors the fit used (qwen3, m2v, bge)")
    ap.add_argument("--bucket", required=True, choices=BUCKETS)
    ap.add_argument("--no-collapse", action="store_true", help="label with the --no-collapse check fit")
    ap.add_argument("--workers", type=int, default=2, help="processes, each holding the model (set by RAM)")
    args = ap.parse_args()
    start = time.monotonic()                     # wall clock for the run, printed at the end

    out, dedup = args.out, f"{args.out}/dedup"
    fit_dir = f"{out}/topics/{args.model}/{args.bucket}" + ("_nocollapse" if args.no_collapse else "")
    if not os.path.exists(f"{fit_dir}/info.json"):
        sys.exit(f"no fit in {fit_dir}; run fit_topics.py first")
    dest = f"{fit_dir}/assigned"
    os.makedirs(dest, exist_ok=True)

    pages = np.load(f"{dedup}/pages.npy")
    with open(f"{dedup}/files.txt", encoding="utf-8") as f:
        names = f.read().split()
    pages = pages[pages["bucket"] == BUCKETS.index(args.bucket)]

    # bucket ids split by file: sort by file, then cut where the file number changes
    by_file = pages[np.argsort(pages["file"], kind="stable")]
    files, cuts = np.unique(by_file["file"], return_index=True)
    groups = dict(zip(files.tolist(), np.split(by_file["id"], cuts[1:])))
    todo = [i for i in range(len(names)) if not os.path.exists(f"{dest}/{names[i]}.npz")]
    missing = [names[i] for i in todo
               if not os.path.exists(f"{out}/embeddings/{args.model}/{names[i]}.npy")]
    if missing:
        sys.exit(f"{len(missing):,} files have no {args.model} vectors; finish embed.py first")
    console.print(f"{len(pages):,} {args.bucket} pages in {len(files):,} files; "
                  f"{len(names) - len(todo):,} files already labeled")

    # every file gets an output, even one with no bucket pages, so it counts as done
    empty = np.empty(0, np.int64)
    results = run_pool("files", assign_file,
                       [(out, args.model, names[i], groups.get(i, empty), dest) for i in todo],
                       args.workers, initializer=init, initargs=(fit_dir,))
    n = sum(r[0] for r in results)
    moved = sum(r[1] for r in results)
    console.print(f"\nthis run: {n:,} pages labeled; {moved:,} ({moved / n if n else 0:.1%}) "
                  f"were outliers moved to their closest topic -> {dest}")
    console.print(f"took {took(start)}")


if __name__ == "__main__":
    main()