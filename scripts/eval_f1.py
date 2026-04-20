import json
import os
import re
import string
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict, Counter

# ---------------------------
# CLI arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Evaluate predictions using F1 score")
parser.add_argument("--predictions", type=str, required=True, help="Path to the predictions file (JSON)")
args = parser.parse_args()

# Assume args.predictions is a string path, possibly without ".json"
pred_path = Path(args.predictions)

if not pred_path.suffix == ".json":
    pred_path = pred_path.with_suffix(".json")

PREDICTIONS_PATH = pred_path
EVAL_OUTPUT_PATH = PREDICTIONS_PATH.with_name(f"{PREDICTIONS_PATH.stem}_eval_f1.json")

# ---------------------------
# Normalization and F1 functions
# ---------------------------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    if prediction is None or ground_truth is None:
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------
# Format validation
# ---------------------------
def is_valid_sequence(text):
    content = text

    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        if content.count(f"<{tag}>") != content.count(f"</{tag}>"):
            return False, f"Unbalanced <{tag}> tags"

    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    state = "start"

    for part in parts:
        if not part.strip():
            continue

        if re.match(r"</?(?:think|search|information|answer)>", part):
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            # added the possibility to have <answer> tag after </information>
            elif part == "<answer>" and state in ["after_think", "information"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                continue
            if part.strip():
                return False, f"Unexpected content '{part.strip()}' in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"



# ---------------------------
# Load predictions
# ---------------------------
if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_PATH}")

with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
    predictions = json.load(f)

print(f"\nEvaluating {len(predictions)} predictions from {PREDICTIONS_PATH.name}...")

results = []
format_scores = []
search_counts = []

for pred in tqdm(predictions):
    data_source = pred["data_source"]
    question = pred["question"]
    golden_answers = pred["golden_answers"]
    
    predicted_answer = pred["predicted_answer"]  # Used for F1
    raw_predicted_answer = pred.get("raw_predicted_answer", "")  # Used for format & search count

    golden_answer = golden_answers[0] if golden_answers else ""
    f1 = f1_score(predicted_answer, golden_answer)

    # Compute format validity and search count from raw_predicted_answer
    is_valid, format_msg = is_valid_sequence(raw_predicted_answer)
    format_scores.append(1 if is_valid else 0)

    search_blocks = re.findall(r"<search>.*?</search>", raw_predicted_answer, re.DOTALL)
    search_count = len(search_blocks)
    search_counts.append(search_count)

    results.append({
        "data_source": data_source,
        "question": question,
        "golden_answer": golden_answer,
        "predicted_answer": predicted_answer,
        "f1_score": f1,
        "format_valid": is_valid,
        "format_reason": format_msg,
        "search_count": search_count
    })

# ---------------------------
# Save output
# ---------------------------
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=2, ensure_ascii=False)

mean_format_accuracy = sum(format_scores) / len(format_scores)
mean_search_count = sum(search_counts) / len(search_counts)

print(f"✅ Saved detailed evaluation to {EVAL_OUTPUT_PATH}")

# add _eval_f1.json suffix to the output file
PREDICTIONS_PATH = PREDICTIONS_PATH.with_name(f"{PREDICTIONS_PATH.stem}_eval_f1.json")
# PREDICTIONS_PATH = Path(args.predictions)

# ---------------------------
# Load data
# ---------------------------
if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_PATH}")

with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
    predictions = json.load(f)

# ---------------------------
# Aggregate per dataset
# ---------------------------
dataset_stats = defaultdict(lambda: {
    "total_f1": 0.0,
    "total_format_valid": 0,
    "total_search": 0,
    "count": 0
})

for item in predictions:
    dataset = item.get("data_source", "all")
    f1 = item.get("f1_score")
    format_valid = item.get("format_valid")
    search_count = item.get("search_count", 0)

    dataset_stats[dataset]["total_f1"] += f1 if f1 is not None else 0.0
    dataset_stats[dataset]["total_format_valid"] += int(bool(format_valid))
    dataset_stats[dataset]["total_search"] += search_count
    dataset_stats[dataset]["count"] += 1

# ---------------------------
# Report
# ---------------------------
print(f"| Dataset | Examples | Avg F1 | Format Accuracy | Avg #Search | File |")
print(f"|---------|----------|--------|------------------|--------------|------|")

meta_f1s, meta_format_accs, meta_search_avgs = [], [], []

for dataset, stats in dataset_stats.items():
    count = stats["count"]
    avg_f1 = stats["total_f1"] / count if count > 0 else 0.0
    format_acc = stats["total_format_valid"] / count if count > 0 else 0.0
    avg_search = stats["total_search"] / count if count > 0 else 0.0

    print(f"| {dataset} | {count} | {avg_f1:.4f} | {format_acc:.3f} | {avg_search:.2f} | {PREDICTIONS_PATH.name} |")

    meta_f1s.append(avg_f1)
    meta_format_accs.append(format_acc)
    meta_search_avgs.append(avg_search)

if meta_f1s:
    meta_avg_f1 = sum(meta_f1s) / len(meta_f1s)
    meta_format = sum(meta_format_accs) / len(meta_format_accs)
    meta_search = sum(meta_search_avgs) / len(meta_search_avgs)

    print(f"\n| **Meta-Average** | — | **{meta_avg_f1:.4f}** | **{meta_format:.3f}** | **{meta_search:.2f}** | {PREDICTIONS_PATH.name} |")
