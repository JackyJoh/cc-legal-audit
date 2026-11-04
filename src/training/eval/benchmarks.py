"""
Runs the standard benchmarks on one trained model with lm-evaluation-harness.

Every task is scored by how likely the model finds each answer, so the models
don't need to be chatbots. NUM_FEWSHOT is the same for every model.

Called from evaluate.py. Needs the model's folder, for example
data/source/training/runs/0.9_seed1/final or .../checkpoint-15000.
"""
import lm_eval
from lm_eval.models.huggingface import HFLM

# checkpoints don't save a tokenizer, so it's always loaded from here
TOKENIZER = "HuggingFaceTB/SmolLM2-360M"
TASKS = ["hellaswag", "winogrande", "lambada_openai", "arc_easy", "sciq",
         "boolq", "piqa", "siqa", "openbookqa"]
NUM_FEWSHOT = 0     # example questions shown to the model before each question


# runs the tasks on one model; returns lm-eval's results, one entry per task.
# limit is the number of questions per task, None for all of them
def run_benchmarks(model_path, tasks=TASKS, batch_size=64, limit=None):
    lm = HFLM(pretrained=model_path, tokenizer=TOKENIZER, batch_size=batch_size,
              dtype="bfloat16")
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=NUM_FEWSHOT,
                                      limit=limit, log_samples=False)
    return results["results"]


# pulls each task's scores out of run_benchmarks' results, as
# scores[task][metric] = (value, standard error)
def get_scores(results):
    scores = {}
    for task, metrics in results.items():
        scores[task] = {}
        for name, value in metrics.items():
            if name == "alias" or name.endswith(",stderr"):
                continue
            scores[task][name] = (value, metrics[name + ",stderr"])
    return scores
