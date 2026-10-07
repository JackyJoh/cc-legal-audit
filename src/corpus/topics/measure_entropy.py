"""
Turns the topic labels into the study's numbers: how much each bucket's topic
diversity drops at each dedup threshold, beyond what removing the same number
of pages at random would do, and whether legal drops more than general. Runs
once, after assign_topics.py has labeled the buckets.

Per bucket, threshold (0.6-0.9) and topic level (k, k/4, k/16):

  effective topics  2 ** the Shannon entropy (bits) of the topic counts: how
                    many equal-sized topics the spread is worth. Before dedup
                    (every labeled page) and after (pages not removed)
  change            after vs before, as a %
  random            the same change when the same number of pages is removed
                    at random (averaged over --random-draws tries)
  excess            change minus random: the drop dedup causes beyond chance
  gap               legal excess minus general excess: the headline
  topics lost       share of topics losing at least 50% ("hit hard") and at
                    least 90% ("gone") of their pages, for dedup and random
  removed           share of the bucket's pages dedup removes

The same within length bins (word count: under 100, then 100-199, 200-399,
doubling; the top bins merged until each holds --min-bin pages of the smaller
bucket measured), to show whether length explains a gap.

Error bars: --rounds bootstrap rounds. Each round redraws the pages/ files
with replacement (one draw, shared by both buckets), re-counts from the fixed
labels and removals, and redraws the random removal; the 95% range is the
middle 95% of rounds. Nothing is refit or re-deduped.

Curve: change, random and excess at every merge level from k down to 10
topics, for the figure.

Measures whichever buckets have labels, so it runs on general alone; the gap
needs both. With --no-collapse, legal is measured with the --no-collapse
check fit. One process: everything is counting.

Usage:
  python src/corpus/topics/measure_entropy.py --out data/source --model qwen3 [--no-collapse]

Reads (--out):
  dedup/pages.npy, dedup/files.txt          from score_pairs.py
  dedup/removed_<t>.npy                     from remove_duplicates.py
  topics/<model>/<bucket>/                  info.json, levels.npy, tree.npy
                                            (fit_topics.py) and assigned/
                                            (assign_topics.py)

Prints tables: the headline per topic level, the gap, topics lost, and length
bins.

Outputs (--out/topics/<model>/results/, or results_nocollapse/):
  results.csv   every number, per bucket, level, bin ("all" = no split) and
                threshold, with the excess's 95% range
  gaps.csv      the gap and its 95% range, per level, bin and threshold
  curve.csv     change, random and excess per bucket and threshold at every
                topic count
  bins.csv      each length bin's word range and pages per bucket
  info.json     buckets, pages, topic counts, outlier shares, settings
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
from rich.table import Table
from scipy.cluster.hierarchy import fcluster

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dedup"))
from score_pairs import BUCKETS, console, took  # noqa: E402

THRESHOLDS   = (0.6, 0.7, 0.8, 0.9)
LEVELS       = ("k", "k/4", "k/16")
FIRST_EDGE   = 100              # length bins: under 100 words, then doubling
MIN_BIN      = 1_000
ROUNDS       = 1_000
RANDOM_DRAWS = 100              # random removals averaged for the main numbers
CURVE_FLOOR  = 10
CURVE_DRAWS  = 20
CHUNK        = 100              # bootstrap rounds counted at once
SEED         = 42


# effective topics and bits of topic counts (last axis = topics)
def eff(counts):
    total = counts.sum(-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros(counts.shape), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = -np.where(p > 0, p * np.log2(p), 0).sum(-1)
    return 2 ** bits, bits


# share of topics (present before) left with at most half, and at most a tenth,
# of their pages; before (..., K), after (..., T, K)
def lost(before, after):
    b = before[..., None, :]
    present = b > 0
    left = np.divide(after, b, out=np.ones(np.broadcast_shapes(after.shape, b.shape)), where=present)
    n = np.maximum(present.sum(-1), 1)
    return ((left <= 0.5) & present).sum(-1) / n, ((left <= 0.1) & present).sum(-1) / n


# one bucket's labels, lengths, files and removals, lined up with its pages; (data, None) or (None, why)
def load_bucket(out, model, bucket, pages, removed, n_files, check):
    fit = f"{out}/topics/{model}/{bucket}" + ("_nocollapse" if check else "")
    if not os.path.exists(f"{fit}/info.json"):
        return None, f"no fit in {fit}"
    paths = [p for p in glob.glob(f"{fit}/assigned/*.npz") if not p.endswith(".tmp.npz")]
    if len(paths) < n_files:
        return None, f"{n_files - len(paths):,} of {n_files:,} files not labeled yet in {fit}/assigned"
    parts = [np.load(p) for p in paths]
    ids = np.concatenate([p["id"] for p in parts])
    order = np.argsort(ids)
    ids = ids[order]
    topic = np.concatenate([p["topic"] for p in parts])[order]
    outlier = np.concatenate([p["outlier"] for p in parts])[order]

    mine = pages[pages["bucket"] == BUCKETS.index(bucket)]
    if not len(mine) or not len(ids):
        return None, "no labeled pages"
    k = np.minimum(np.searchsorted(ids, mine["id"]), len(ids) - 1)
    if not (ids[k] == mine["id"]).all():
        return None, f"{int((ids[k] != mine['id']).sum()):,} pages have no label; rerun assign_topics.py"
    with open(f"{fit}/info.json", encoding="utf-8") as f:
        info = json.load(f)
    return {"n": len(mine), "file": mine["file"].astype(np.int64), "words": mine["words"],
            "topic": topic[k].astype(np.int64), "outlier_share": float(outlier[k].mean()),
            "removed": np.stack([np.isin(mine["id"], r) for r in removed]),
            "sizes": (info["k"], info["k4"], info["k16"]),
            "levels": np.load(f"{fit}/levels.npy"), "tree": np.load(f"{fit}/tree.npy")}, None


# length bins shared by every bucket: under 100 words, then doubling, merged from
# the top down until each holds min_bin pages of the smaller bucket. Returns
# (raw edges, raw bin -> bin, bin names)
def length_bins(data, min_bin):
    top = max(int(d["words"].max(initial=0)) for d in data.values())
    edges = [FIRST_EDGE]
    while edges[-1] <= top:
        edges.append(edges[-1] * 2)
    ref = min(data.values(), key=lambda d: d["n"])
    counts = np.bincount(np.searchsorted(edges, ref["words"], side="right"), minlength=len(edges) + 1)
    groups, cur, acc = [], [], 0
    for i in reversed(range(len(counts))):
        cur.append(i)
        acc += counts[i]
        if acc >= min_bin:
            groups.append(cur)
            cur, acc = [], 0
    if cur:                                      # short leftovers join the bin above them
        if groups:
            groups[-1] += cur
        else:
            groups.append(cur)
    groups = [sorted(g) for g in reversed(groups)]
    lows = [0] + edges                           # raw bin i: lows[i] up to lows[i + 1] - 1
    to_bin, names = np.empty(len(counts), np.int64), []
    for j, g in enumerate(groups):
        to_bin[g] = j
        lo, hi = lows[g[0]], (lows[g[-1] + 1] if g[-1] + 1 < len(lows) else None)
        names.append("any" if lo == 0 and hi is None else f"<{hi}" if lo == 0
                     else f"{lo}+" if hi is None else f"{lo}-{hi - 1}")
    return edges, to_bin, names


# pages per (file, bin, state, topic), one row per file, so a bootstrap round's
# counts are its file draw times this; bin nb = all bins, state 0 = before
# dedup, state 1 + t = after threshold t
def count_matrix(d, nb, n_files):
    K, T = d["sizes"][0], len(d["removed"])
    M = np.zeros((n_files, nb + 1, 1 + T, K), np.float32)
    for s in range(1 + T):
        keep = np.ones(d["n"], bool) if s == 0 else ~d["removed"][s - 1]
        idx = (d["file"][keep] * nb + d["bin"][keep]) * K + d["topic"][keep]
        M[:, :nb, s] = np.bincount(idx, minlength=n_files * nb * K).reshape(n_files, nb, K)
    M[:, nb] = M[:, :nb].sum(1)
    return M.reshape(n_files, -1)


# the numbers for counts C (rows, bins, states, K): per level a dict of arrays
# shaped (rows, bins[, T]); random removals drawn `draws` times and averaged
def numbers(C, maps, rng, draws):
    before, after = C[:, :, 0], C[:, :, 1:]
    b_int = np.rint(before).astype(np.int64)
    m = b_int.sum(-1)[..., None] - np.rint(after.sum(-1)).astype(np.int64)   # pages removed
    rows, bins, T = m.shape
    rand = np.empty((draws,) + after.shape)
    for r in range(rows):
        for b in range(bins):
            for t in range(T):
                for x in range(draws):
                    gone = rng.multivariate_hypergeometric(b_int[r, b], m[r, b, t]) if m[r, b, t] > 0 else 0
                    rand[x, r, b, t] = b_int[r, b] - gone
    removed = 1 - after.sum(-1) / np.maximum(before.sum(-1)[..., None], 1)

    res = []
    for A in maps:
        bL, aL, rL = before @ A, after @ A, rand @ A
        e_b, bits_b = eff(bL)
        e_a, bits_a = eff(aL)
        e_r, _ = eff(rL)
        change = e_a / e_b[..., None] - 1
        rchange = (e_r / e_b[None, ..., None] - 1).mean(0)
        hit, gone = lost(bL, aL)
        rhit, rgone = lost(bL[None], rL)
        res.append({"removed": removed, "eff_before": e_b, "bits_before": bits_b,
                    "eff_after": e_a, "bits_after": bits_a, "change": change, "random": rchange,
                    "excess": change - rchange, "hit": hit, "gone": gone,
                    "random_hit": rhit.mean(0), "random_gone": rgone.mean(0)})
    return res


# change, random and excess at every topic count from k down to CURVE_FLOOR
def curve(d, C, rng, thresholds):
    K = d["sizes"][0]
    before, after = C[-1, 0], C[-1, 1:]                     # all bins
    b_int = np.rint(before).astype(np.int64)
    m = b_int.sum() - np.rint(after.sum(-1)).astype(np.int64)
    rand = np.stack([[b_int - (rng.multivariate_hypergeometric(b_int, mt) if mt > 0 else 0) for mt in m]
                     for _ in range(CURVE_DRAWS)])          # draws, T, K
    rows, prev = [], None
    for n in range(K, min(CURVE_FLOOR, K) - 1, -1):
        lab = np.arange(K) if n == K else fcluster(d["tree"], n, criterion="maxclust") - 1
        nn = int(lab.max()) + 1
        if nn == prev:
            continue
        prev = nn
        A = np.eye(nn)[lab]
        e_b, _ = eff(before @ A)
        change = eff(after @ A)[0] / e_b - 1
        rchange = (eff(rand @ A)[0] / e_b - 1).mean(0)
        rows += [(nn, t, change[i], rchange[i], change[i] - rchange[i]) for i, t in enumerate(thresholds)]
    return rows


def pct(x):
    return f"{x * 100:+.1f}%"


def span(lo, hi):
    return f"{lo * 100:+.1f} to {hi * 100:+.1f}"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/source", help="source_pages.py's --out folder")
    ap.add_argument("--model", required=True, help="which embed.py vectors the fits used (qwen3, m2v, bge)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    ap.add_argument("--rounds", type=int, default=ROUNDS, help="bootstrap rounds")
    ap.add_argument("--random-draws", type=int, default=RANDOM_DRAWS, help="random removals averaged")
    ap.add_argument("--min-bin", type=int, default=MIN_BIN, help="fewest pages of the smaller bucket per length bin")
    ap.add_argument("--no-collapse", action="store_true", help="measure legal with the --no-collapse check fit")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    start = time.monotonic()                     # wall clock for the run, printed at the end
    rng = np.random.default_rng(args.seed)
    thresholds = sorted(args.thresholds)

    out, dedup = args.out, f"{args.out}/dedup"
    pages = np.load(f"{dedup}/pages.npy")
    with open(f"{dedup}/files.txt", encoding="utf-8") as f:
        n_files = len(f.read().split())
    removed = []
    for t in thresholds:
        if not os.path.exists(f"{dedup}/removed_{t}.npy"):
            sys.exit(f"no {dedup}/removed_{t}.npy; run remove_duplicates.py first")
        removed.append(np.load(f"{dedup}/removed_{t}.npy")["id"])

    console.print("[bold]1. load[/bold]")
    step = time.monotonic()
    data = {}
    for b in BUCKETS:
        d, why = load_bucket(out, args.model, b, pages, removed, n_files, args.no_collapse and b == "legal")
        if d is None:
            console.print(f"  {b}: skipped ({why})")
            continue
        data[b] = d
        console.print(f"  {b}: {d['n']:,} pages, {' -> '.join(f'{s:,}' for s in d['sizes'])} topics, "
                      f"{d['outlier_share']:.1%} started as outliers")
    if not data:
        sys.exit("no bucket has topic labels; run fit_topics.py and assign_topics.py first")
    edges, to_bin, bin_names = length_bins(data, args.min_bin)
    for d in data.values():
        d["bin"] = to_bin[np.searchsorted(edges, d["words"], side="right")]
    slots = bin_names + ["all"]
    nb = len(bin_names)
    console.print(f"  length bins: {', '.join(bin_names)}")
    console.print(f"  took {took(step)}")

    console.print("[bold]2. count[/bold]")
    step = time.monotonic()
    M, maps, main = {}, {}, {}
    for b, d in data.items():
        K = d["sizes"][0]
        M[b] = count_matrix(d, nb, n_files)
        maps[b] = [np.eye(K), np.eye(d["sizes"][1])[d["levels"][:, 0]], np.eye(d["sizes"][2])[d["levels"][:, 1]]]
        C = M[b].sum(0, dtype=np.float64).reshape(1, nb + 1, 1 + len(thresholds), K)
        main[b] = [{key: v[0] for key, v in r.items()} for r in numbers(C, maps[b], rng, args.random_draws)]
    console.print(f"  took {took(step)}")

    console.print(f"[bold]3. bootstrap[/bold] ({args.rounds:,} rounds)")
    step = time.monotonic()
    boot = {b: [[] for _ in LEVELS] for b in data}           # excess per round, per level
    for done in range(0, args.rounds, CHUNK):
        r = min(CHUNK, args.rounds - done)
        W = np.stack([np.bincount(rng.integers(0, n_files, n_files), minlength=n_files)
                      for _ in range(r)]).astype(np.float32)  # times each file is drawn
        for b, d in data.items():
            C = (W @ M[b]).astype(np.float64).reshape(r, nb + 1, 1 + len(thresholds), d["sizes"][0])
            for li, res in enumerate(numbers(C, maps[b], rng, 1)):
                boot[b][li].append(res["excess"])
    boot = {b: [np.concatenate(x) for x in levels] for b, levels in boot.items()}   # rounds, bins, T
    lo = {b: [np.percentile(x, 2.5, axis=0) for x in v] for b, v in boot.items()}
    hi = {b: [np.percentile(x, 97.5, axis=0) for x in v] for b, v in boot.items()}
    both = all(b in data for b in BUCKETS)
    if both:
        gap =[main["legal"][li]["excess"] - main["general"][li]["excess"] for li in range(len(LEVELS))]
        gap_boot = [boot["legal"][li] - boot["general"][li] for li in range(len(LEVELS))]
        gap_lo = [np.percentile(g, 2.5, axis=0) for g in gap_boot]
        gap_hi = [np.percentile(g, 97.5, axis=0) for g in gap_boot]
    console.print(f"  took {took(step)}")

    console.print("[bold]4. curve[/bold]")
    step = time.monotonic()
    curves = {b: curve(d, M[b].sum(0, dtype=np.float64).reshape(nb + 1, 1 + len(thresholds), d["sizes"][0]),
                       rng, thresholds) for b, d in data.items()}
    console.print(f"  took {took(step)}")

    # tables: headline per level, gap, topics lost (level k), length bins (level k)
    a = nb                                       # the "all" slot
    for li, level in enumerate(LEVELS):
        table = Table("threshold", "bucket", "removed", "effective topics", "change", "random", "excess",
                      "excess 95%", title=f"\ntopic level {level}")
        for ti, t in enumerate(thresholds):
            for b in data:
                r = main[b][li]
                table.add_row(f"{t}", b, f"{r['removed'][a, ti]:.1%}",
                              f"{r['eff_before'][a]:,.1f} -> {r['eff_after'][a, ti]:,.1f}",
                              pct(r["change"][a, ti]), pct(r["random"][a, ti]), pct(r["excess"][a, ti]),
                              span(lo[b][li][a, ti], hi[b][li][a, ti]))
        console.print(table)

    if both:
        table = Table("threshold", *(f"gap at {lv} (95%)" for lv in LEVELS),
                      title="\ngap: legal excess minus general excess, in points")
        for ti, t in enumerate(thresholds):
            table.add_row(f"{t}", *(f"{gap[li][a, ti] * 100:+.1f} ({span(gap_lo[li][a, ti], gap_hi[li][a, ti])})"
                                    for li in range(len(LEVELS))))
        console.print(table)

    table = Table("threshold", "bucket", "hit hard", "hit hard (random)", "gone", "gone (random)",
                  title="\ntopics lost at level k: share losing >= 50% (hit hard) and >= 90% (gone) of pages")
    for ti, t in enumerate(thresholds):
        for b in data:
            r = main[b][0]
            table.add_row(f"{t}", b, f"{r['hit'][a, ti]:.1%}", f"{r['random_hit'][a, ti]:.1%}",
                          f"{r['gone'][a, ti]:.1%}", f"{r['random_gone'][a, ti]:.1%}")
    console.print(table)

    cols = ["words", "threshold"]
    for b in data:
        cols += [f"{b} pages", f"{b} removed", f"{b} excess"]
    if both:
        cols.append("gap (95%)")
    table = Table(*cols, title="\nlength bins at level k")
    for j, name in enumerate(bin_names):
        for ti, t in enumerate(thresholds):
            row = [name if ti == 0 else "", f"{t}"]
            for b, d in data.items():
                r = main[b][0]
                row += [f"{int((d['bin'] == j).sum()):,}" if ti == 0 else "", f"{r['removed'][j, ti]:.1%}",
                        pct(r["excess"][j, ti])]
            if both:
                row.append(f"{gap[0][j, ti] * 100:+.1f} ({span(gap_lo[0][j, ti], gap_hi[0][j, ti])})")
            table.add_row(*row)
    console.print(table)

    dest = f"{out}/topics/{args.model}/results" + ("_nocollapse" if args.no_collapse else "")
    os.makedirs(dest, exist_ok=True)
    keys = ["removed", "eff_before", "eff_after", "bits_before", "bits_after", "change", "random", "excess",
            "hit", "gone", "random_hit", "random_gone"]
    rows = []
    for b in data:
        for li, level in enumerate(LEVELS):
            r = main[b][li]
            for j, slot in enumerate(slots):
                for ti, t in enumerate(thresholds):
                    vals = [r[key][j] if r[key].ndim == 1 else r[key][j, ti] for key in keys]
                    rows.append([b, level, slot, t, *(float(v) for v in vals),
                                 float(lo[b][li][j, ti]), float(hi[b][li][j, ti])])
    write_csv(f"{dest}/results.csv", ["bucket", "level", "bin", "threshold", *keys, "excess_lo", "excess_hi"], rows)
    if both:
        write_csv(f"{dest}/gaps.csv", ["level", "bin", "threshold", "gap", "gap_lo", "gap_hi"],
                  [[level, slot, t, float(gap[li][j, ti]), float(gap_lo[li][j, ti]), float(gap_hi[li][j, ti])]
                   for li, level in enumerate(LEVELS) for j, slot in enumerate(slots)
                   for ti, t in enumerate(thresholds)])
    write_csv(f"{dest}/curve.csv", ["bucket", "topics", "threshold", "change", "random", "excess"],
              [[b, *row] for b, rows_b in curves.items() for row in rows_b])
    write_csv(f"{dest}/bins.csv", ["bin", *(f"{b}_pages" for b in data)],
              [[name, *(int((d["bin"] == j).sum()) for d in data.values())] for j, name in enumerate(bin_names)])
    info = {"model": args.model, "no_collapse": args.no_collapse, "thresholds": thresholds,
            "rounds": args.rounds, "random_draws": args.random_draws, "min_bin": args.min_bin, "seed": args.seed,
            "buckets": {b: {"pages": d["n"], "topics": list(d["sizes"]), "outlier_share": d["outlier_share"]}
                        for b, d in data.items()}}
    with open(f"{dest}/info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    console.print(f"\nresults -> {dest}")
    console.print(f"took {took(start)}")


if __name__ == "__main__":
    main()