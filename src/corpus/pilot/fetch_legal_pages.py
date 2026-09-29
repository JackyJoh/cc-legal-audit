"""
Samples candidate legal pages for the MinHash pilot, in the same site mix as
open-web legal.

Sites and their shares come from Laya v2's open-web keeps (p >= 0.85). Each
site's target is --total x its share; --overdraw x that many random pages are
drawn, since not every page on a site is legal. fetch_control_pool.py trims
Laya's keeps back to each target.

Writes data/pilot/legal_pool.jsonl (url, site, target, WARC pointers).

Then:
  python src/common/fetch_warc_text.py --input data/pilot/legal_pool.jsonl
      --output data/pilot/legal_pool_text.jsonl
  C:\\projects\\laya-env\\Scripts\\python.exe src/classifier/largeModel/score_laya.py
      --model models/laya-legal --max-len 1024 --batch-size 2
      --input data/pilot/legal_pool_text.jsonl --output data/pilot/laya_legal_scores.jsonl
"""
import argparse
import os
from collections import Counter

from dotenv import load_dotenv

from sampling import OUT_DIR, load_jsonl, one_capture_per_url, pages_sql, site_of, write_jsonl
from athena import client, run_query  # noqa: E402  (path set by sampling)

load_dotenv()

OPENWEB_SCORES = "data/candidates/laya_openweb_scores.jsonl"
LAYA_MODEL     = os.path.normpath("models/laya-legal")
LAYA_MAX_LEN   = 1024
LAYA_CUT       = 0.85
OUTPUT         = f"{OUT_DIR}/legal_pool.jsonl"


def openweb_site_shares():
    """Site -> share of Laya's open-web keeps."""
    keeps = {r["url"] for r in load_jsonl(OPENWEB_SCORES)
             if os.path.normpath(str(r.get("model"))) == LAYA_MODEL
             and r.get("max_len") == LAYA_MAX_LEN and r["p_legal"] >= LAYA_CUT}
    counts = Counter(site_of(u) for u in keeps)
    return {s: n / len(keeps) for s, n in counts.items()}, len(keeps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--total", type=int, default=2000, help="legal pages wanted after Laya")
    ap.add_argument("--overdraw", type=float, default=3.0, help="pages drawn per page wanted")
    args = ap.parse_args()

    shares, n_keeps = openweb_site_shares()
    target = {s: max(1, round(args.total * p)) for s, p in shares.items()}
    draw = {s: round(t * args.overdraw) for s, t in target.items()}
    print(f"{n_keeps} open-web keeps across {len(shares)} sites; "
          f"target {sum(target.values())}, drawing up to {sum(draw.values())}")

    rows, _ = run_query(client(), pages_sql(draw))
    pool = [{**r, "target": target[r["site"]]} for r in one_capture_per_url(rows)]
    write_jsonl(OUTPUT, pool)

    got = Counter(r["site"] for r in pool)
    print(f"\n  {'site':<32}{'share':>7}{'target':>8}{'drawn':>7}")
    for s in sorted(shares, key=lambda s: -shares[s]):
        flag = "   SHORT" if got[s] < target[s] else ""
        print(f"  {s[:31]:<32}{shares[s]:>7.1%}{target[s]:>8}{got[s]:>7}{flag}")
    print(f"\n{len(pool)} pages -> {OUTPUT}")


if __name__ == "__main__":
    main()