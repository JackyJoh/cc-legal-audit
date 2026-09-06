"""
Draws a uniform random sample of URLs from a Common Crawl snapshot.

Usage: python src/samples/fetch_candidate_urls.py [--n 600000] [--output F]

Pure sourcing, no classification. Two things depend on this sample being
genuinely uniform: the base rate of legal pages in the crawl, and every
deployment precision number measured against it. A skewed sample makes both
unsupported without looking wrong.

How the sample is drawn:

  1. Count the rows in the snapshot matching the filter, so the sampling rate
     can be set from the real population size instead of guessed at.
  2. TABLESAMPLE BERNOULLI(p), with p set to land near the requested count.
     Bernoulli sampling rolls per row independently, so every row has the
     same chance of selection and the result is uniform.

There is deliberately no LIMIT. An earlier version used BERNOULLI(1) with
LIMIT 600000, which is not a uniform sample: Athena reads the index files in
parallel and LIMIT stops at the first N rows returned, so the output is
whichever files answered fastest. Those files group by domain and crawl
segment, so the result skewed toward part of the crawl. Shuffling afterwards
cannot fix that, since the missing rows were never returned. Because p is set
from a count rather than fixed, the row count varies run to run by a percent
or so, which is what uniform sampling looks like.

Each row carries its WARC pointer (archive file, byte offset, length)
alongside the URL. Those columns are free on a query that is already scanning
the partition, and they let src/corpus/fetch_warc_text.py fetch page text for
this sample without a second index lookup.

The content_languages = 'eng' filter is an exact match, so it takes only
pages the crawl tagged as monolingual English and drops multilingual ones
like 'eng,fra'. That is a real skew in what this sample represents, kept
because changing it would break comparability with everything already labeled
from earlier pulls.

Cost: about $0.50 per run, one partition scan, independent of how many rows
are requested.
"""
import argparse
import json
import os

from dotenv import load_dotenv

from athena import client, run_query

load_dotenv()

SNAPSHOT = "CC-MAIN-2026-12"
DEFAULT_N = 600_000
DEFAULT_OUTPUT = "data/candidates/raw_pool.jsonl"
# what a sampled row carries: the URL plus everything needed to fetch its
# page text later without asking the index again
COLUMNS = ["url", "warc_filename", "warc_record_offset", "warc_record_length",
           "fetch_status", "content_mime_detected", "content_languages"]
FILTER = f"""crawl = '{SNAPSHOT}'
      AND subset = 'warc'
      AND content_languages = 'eng'"""


def population_size(athena):
    """Rows matching the filter, so the sampling rate is set from the actual
    population rather than an assumption about crawl size."""
    rows, _ = run_query(athena, f"SELECT count(*) AS n FROM ccindex WHERE {FILTER}")
    return int(rows[0]["n"])


def sample(athena, percent):
    sql = f"""
        SELECT {', '.join(COLUMNS)}
        FROM ccindex TABLESAMPLE BERNOULLI({percent})
        WHERE {FILTER}
    """
    rows, _ = run_query(athena, sql)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="target row count (the actual count varies slightly, "
                         "since the sampling rate is a probability)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    athena = client()
    print(f"Snapshot : {SNAPSHOT}")

    print("Counting matching rows...")
    total = population_size(athena)
    percent = min(100.0, args.n / total * 100)
    print(f"Population: {total:,} rows")
    print(f"Sampling  : BERNOULLI({percent:.6f}) for ~{args.n:,} rows")

    rows = sample(athena, percent)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Sampled  : {len(rows):,} URLs ({len(rows) / total:.4%} of the population)")
    print(f"Written  : {args.output}")


if __name__ == "__main__":
    main()
