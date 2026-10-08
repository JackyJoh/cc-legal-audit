"""
Learns one bucket's topics with BERTopic, once, from a sample of its pages, and
groups similar topics into bigger ones. Labels no pages itself;
assign_topics.py gives every page its topic. Runs once, after dedup.

Four steps:

  1. sample   the bucket's pages, minus those removed at 0.6 (so a template
              copied 3,000 times counts once and can't become a topic of its
              own), then min(that, --max-fit) of them, drawn with --seed
  2. read     each sampled page's vector (embeddings/) and text (pages/; the
              text only names the topics by their common words)
  3. fit      BERTopic on the saved vectors: squash them (UMAP), find clumps
              (HDBSCAN), name each clump. A clump needs at least 0.05% of the
              sample (and at least 10 pages) to count as a topic; pages in no
              clump are outliers, handled by assign_topics.py
  4. group    each topic's average vector, then merge two topics again and
              again until one is left, each time the pair that keeps groups
              tightest (Ward), so groups grow evenly. Cutting that order at k/4 and
              k/16 topics (the smaller at least 10) gives the bigger topics;
              the full order is kept for the curve across every size

With --no-collapse, step 1 keeps the pages removed at 0.6: the check that
copies didn't shape the topics (legal only). Its output goes to a separate
folder.

Workers:

  read   a pool of processes (--workers), each taking one pages/ file at a
         time and pulling out its sampled pages' vectors and text
  fit    one process. With a fixed seed UMAP runs on one core (hours at 1M
         pages); HDBSCAN uses every core. --gpu runs both on the GPU

A bucket that already has a finished fit is skipped; delete its folder to fit
again. The output is written to a temp folder and swapped in at the end, so a
crash never leaves a half-written fit.

Runs in the laya environment, plus bertopic (and cuml for --gpu), not the
project venv.

Usage:
  C:\\projects\\laya-env\\Scripts\\python.exe src/corpus/topics/fit_topics.py --out data/source --bucket legal

Reads (--out):
  dedup/pages.npy, dedup/files.txt   from score_pairs.py
  dedup/removed_0.6.npy              from remove_duplicates.py
  fingerprints/<file>.npz            each page's id, in pages/ order
  embeddings/<model>/<file>.npy      from embed.py
  pages/<file>.jsonl.gz              text of the sampled pages

Outputs (--out/topics/<model>/<bucket>/, or <bucket>_nocollapse/):
  model/            the fitted BERTopic; assign_topics.py labels pages with it
  levels.npy        one row per topic (row = topic number): the bigger topic it
                    belongs to at k/4, then at k/16
  tree.npy          the full merge order (scipy linkage), for the curve
  fit_ids.npy       ids of the pages the fit used
  topic_info.csv    each topic's page count and top words, to read by eye
  info.json         fit size, min topic size, k, k/4, k/16, outlier share

Next:
  python src/corpus/topics/assign_topics.py --out data/source --bucket legal
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dedup"))
from score_pairs import BUCKETS, console, run_pool, save, took  # noqa: E402

COLLAPSE    = 0.6               # threshold whose removed pages leave the fit sample
MAX_FIT     = 1_000_000
MIN_SHARE   = 0.0005            # min topic size, as a share of the fit size
MIN_TOPIC   = 10                # ...and never below this many pages
MIN_LEVEL   = 10                # the k/16 level never has fewer topics than this
SEED        = 42
MAX_WORDS   = 50_000            # vocabulary kept for naming topics


# one pages/ file's sampled pages: (ids, vectors, texts), in file order. The
# vector and text rows line up with fingerprints/ because all three keep
# pages/ order; a URL seen twice in a file takes its first row
def read_file(out, model, name, wanted):
    ids = np.load(f"{out}/fingerprints/{name}.npz")["id"]
    vecs = np.load(f"{out}/embeddings/{model}/{name}.npy", mmap_mode="r")
    if len(vecs) != len(ids):
        raise ValueError(f"{name}: {len(ids)} pages but {len(vecs)} vectors; "
                         f"delete embeddings/{model}/{name}.npy and rerun embed.py")
    _, first = np.unique(ids, return_index=True)
    rows = np.sort(first[np.isin(ids[first], wanted)])
    want, texts, i = set(rows.tolist()), [], 0
    with gzip.open(f"{out}/pages/{name}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                if i in want:
                    texts.append(json.loads(line)["text"])
                i += 1
    return ids[rows], np.asarray(vecs[rows], np.float32), texts


# step 3: the BERTopic model, fitted; returns it and each sampled page's topic (-1 = outlier)
def fit(texts, vecs, min_topic, seed, gpu):
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    if gpu:
        from cuml.cluster import HDBSCAN
        from cuml.manifold import UMAP
    else:
        from hdbscan import HDBSCAN
        from umap import UMAP
    # a fixed random_state makes UMAP repeatable, at the cost of running it on one core
    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    clumps = HDBSCAN(min_cluster_size=min_topic, metric="euclidean", prediction_data=True)
    words = CountVectorizer(stop_words="english", max_features=MAX_WORDS)
    model = BERTopic(umap_model=umap, hdbscan_model=clumps, vectorizer_model=words, verbose=True)
    topics, _ = model.fit_transform(texts, embeddings=vecs)
    return model, np.asarray(topics)


# step 4: each topic's average vector, the merge order, and each topic's group
# at k/4 and k/16 (0-based); returns (levels, tree, sizes of the two levels)
def group(vecs, topics, k):
    centers = np.stack([vecs[topics == t].mean(axis=0) for t in range(k)])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Ward: each merge keeps groups as tight as possible, so groups grow evenly
    # instead of one big group absorbing its neighbours (unit-length vectors, so
    # straight-line distance ranks like cosine)
    tree = linkage(centers, method="ward")
    coarse = min(max(round(k / 16), MIN_LEVEL), k)
    mid = min(max(round(k / 4), coarse), k)
    levels = np.stack([fcluster(tree, n, criterion="maxclust") - 1 for n in (mid, coarse)], axis=1)
    return levels.astype(np.int32), tree, (mid, coarse)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/source", help="source_pages.py's --out folder")
    ap.add_argument("--model", default="bge", help="which embed.py vectors to use")
    ap.add_argument("--bucket", required=True, choices=BUCKETS)
    ap.add_argument("--max-fit", type=int, default=MAX_FIT, help="most pages the fit uses")
    ap.add_argument("--no-collapse", action="store_true", help="keep pages removed at 0.6 in the sample (the check)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes reading pages/ files")
    ap.add_argument("--gpu", action="store_true", help="run UMAP and HDBSCAN on the GPU (needs cuml)")
    args = ap.parse_args()
    start = time.monotonic()                     # wall clock for the run, printed at the end

    out, dedup = args.out, f"{args.out}/dedup"
    dest = f"{out}/topics/{args.model}/{args.bucket}" + ("_nocollapse" if args.no_collapse else "")
    if os.path.exists(f"{dest}/info.json"):
        sys.exit(f"{dest} already has a fit; delete it to fit again")
    for need, by in ((f"{dedup}/pages.npy", "score_pairs.py"),
                     (f"{dedup}/removed_{COLLAPSE}.npy", "remove_duplicates.py")):
        if not os.path.exists(need):
            sys.exit(f"no {need}; run {by} first")

    console.print("[bold]1. sample[/bold]")
    step = time.monotonic()
    pages = np.load(f"{dedup}/pages.npy")
    with open(f"{dedup}/files.txt", encoding="utf-8") as f:
        names = f.read().split()
    pages = pages[pages["bucket"] == BUCKETS.index(args.bucket)]
    in_bucket = len(pages)
    if not args.no_collapse:
        pages = pages[~np.isin(pages["id"], np.load(f"{dedup}/removed_{COLLAPSE}.npy")["id"])]
    rng = np.random.default_rng(args.seed)
    pick = pages[np.sort(rng.choice(len(pages), min(len(pages), args.max_fit), replace=False))]
    min_topic = max(MIN_TOPIC, round(MIN_SHARE * len(pick)))
    console.print(f"  {in_bucket:,} {args.bucket} pages, {len(pages):,} after the {COLLAPSE} collapse"
                  if not args.no_collapse else f"  {in_bucket:,} {args.bucket} pages, no collapse")
    console.print(f"  {len(pick):,} sampled; a topic needs at least {min_topic:,} of them")
    missing = [names[i] for i in np.unique(pick["file"])
               if not os.path.exists(f"{out}/embeddings/{args.model}/{names[i]}.npy")]
    if missing:
        sys.exit(f"  {len(missing):,} files have no {args.model} vectors; finish embed.py first")
    console.print(f"  took {took(step)}")

    console.print("[bold]2. read[/bold]")
    step = time.monotonic()
    # sampled ids split by file: sort by file, then cut where the file number changes
    by_file = pick[np.argsort(pick["file"], kind="stable")]
    files, cuts = np.unique(by_file["file"], return_index=True)
    groups = np.split(by_file["id"], cuts[1:])
    parts = run_pool("files", read_file,
                     [(out, args.model, names[i], g) for i, g in zip(files, groups)], args.workers)
    ids   = np.concatenate([p[0] for p in parts])
    vecs  = np.concatenate([p[1] for p in parts])
    texts = [t for p in parts for t in p[2]]
    console.print(f"  {len(ids):,} vectors and texts from {len(files):,} files")
    console.print(f"  took {took(step)}")

    console.print("[bold]3. fit[/bold]")
    step = time.monotonic()
    model, topics = fit(texts, vecs, min_topic, args.seed, args.gpu)
    k = int(topics.max()) + 1
    if k < 2:
        sys.exit(f"  only {k} topic found; too few pages to fit {args.bucket}")
    outliers = float((topics == -1).mean())
    console.print(f"  {k:,} topics; {outliers:.1%} of sampled pages in none (outliers)")
    console.print(f"  took {took(step)}")

    console.print("[bold]4. group[/bold]")
    step = time.monotonic()
    levels, tree, (mid, coarse) = group(vecs, topics, k)
    console.print(f"  {k:,} topics -> {mid:,} (k/4) -> {coarse:,} (k/16)")
    console.print(f"  took {took(step)}")

    # everything into a temp folder, then swapped in, so a crash leaves no half-written fit
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    model.save(f"{tmp}/model", serialization="pickle", save_embedding_model=False)
    save(f"{tmp}/levels.npy", levels)
    save(f"{tmp}/tree.npy", tree)
    save(f"{tmp}/fit_ids.npy", ids)
    model.get_topic_info()[["Topic", "Count", "Name"]].to_csv(f"{tmp}/topic_info.csv", index=False)
    info = {"model": args.model, "bucket": args.bucket, "collapsed": not args.no_collapse,
            "pages_in_bucket": in_bucket, "pages_after_collapse": len(pages), "fit_size": len(ids),
            "min_topic_size": min_topic, "k": k, "k4": mid, "k16": coarse,
            "outlier_share": outliers, "seed": args.seed}
    with open(f"{tmp}/info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)

    console.print(f"\n{args.bucket}: {k:,} topics from {len(ids):,} pages -> {dest}")
    console.print(f"took {took(start)}")


if __name__ == "__main__":
    main()