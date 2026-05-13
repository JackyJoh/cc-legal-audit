from datasets import load_dataset
from URL_Classifier import classify
import random


ds = load_dataset("pile-of-law/pile-of-law", "canadian_decisions", split="train")
#olc_memos

def Recall():

    random.seed(42)
    sample = random.sample(range(len(ds)), 50)
    falseNegatives = set()
    truePositives = set()
    good = 0
    total = 0

    for index in sample:
        url = ds[index]['url']
        if (classify(url)):
            truePositives.add(url)
            good+=1
        else:
            falseNegatives.add(url)
        total+=1
    print(f"False Negatives: {falseNegatives}")
    print(f"\n\nTrue Positives: {truePositives}")

    return good, total

if __name__ == "__main__":
    good, total = Recall()

    print(f"{good} out of {total} docs were correctly labeled as legal")
    print(f"Ratio of: {good/total}")

