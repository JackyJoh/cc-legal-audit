"""
Runs tests on one trained model and saves the scores.

Every test is a function in TESTS that takes the model's folder and the command
line arguments, and returns scores[group][metric] = (value, standard error). A
group is whatever the test splits its scores by, such as a benchmark task. To
add a test, write its function in its own file and add it to TESTS; printing
and saving already work for it.

--limit runs only that many questions per task, to check everything works
before a full run. Those results get _limit<N> in their file name, so they can't
overwrite a full run's.

--tokenizer is only for a model trained with a different tokenizer than ours,
such as Pythia-410M, the reference model. --model can also be a Hugging Face name.

Runs in the eval environment (~/venvs/eval on HiPerGator), on a GPU, from the
repo folder.

Usage:
  python -u src/training/eval/evaluate.py --model data/source/training/runs/0.9_seed1/final
      [--tokenizer EleutherAI/pythia-410m] [--test benchmarks] [--batch-size 64] [--limit 10]

Outputs (data/source/training/evals/):
  <run>_<checkpoint>_<test>.json   model, tokenizer, test, limit, batch size, scores
"""
import argparse
import json
import os

from benchmarks import TOKENIZER, get_scores, run_benchmarks

RUNS = "data/source/training/runs"
OUT = "data/source/training/evals"


def benchmarks(model_path, args):
    return get_scores(run_benchmarks(model_path, tokenizer=args.tokenizer,
                                     batch_size=args.batch_size, limit=args.limit))


TESTS = {
    "benchmarks": benchmarks,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the model's folder")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--test", nargs="+", choices=TESTS, default=list(TESTS),
                    help="tests to run (default all)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, help="questions per task (default all)")
    args = ap.parse_args()

    # 0.9_seed1_final for data/source/training/runs/0.9_seed1/final
    model_name = args.model.rstrip("/").removeprefix(f"{RUNS}/").replace("/", "_")
    os.makedirs(OUT, exist_ok=True)

    for test in args.test:
        scores = TESTS[test](args.model, args)

        print(f"\n{model_name}, {test}")
        for group, metrics in scores.items():
            for metric, (value, stderr) in metrics.items():
                error = f"{stderr:.4f}" if isinstance(stderr, float) else stderr
                print(f"  {group:20} {metric:12} {value:.4f}  +/- {error}")

        suffix = f"_limit{args.limit}" if args.limit else ""
        path = f"{OUT}/{model_name}_{test}{suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "tokenizer": args.tokenizer, "test": test,
                       "limit": args.limit, "batch_size": args.batch_size,
                       "scores": scores}, f, indent=2)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
