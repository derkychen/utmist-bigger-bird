"""Long-context benchmark loaders for exp 20.

Every row is a 4-way multiple-choice item split into `prefix` (instructions), `context` (the long document /
repository, the only part that is truncated) and `suffix` (question + choices), plus the gold letter.

  lbv2        LongBench v2 (Bai et al., 2024), 503 items, 6 domains incl. Code Repository Understanding.
              Official no-CoT template (prompts/0shot.txt).
  codedebug   InfiniteBench Code.Debug (Zhang et al., 2024), 394 items: find the function with a planted bug
              in a real Python repo (~180K+ tokens). Official template (InfiniteBench/src/prompt.py).
  lcqa32k / lcqa64k / lcqa128k
              LongCodeBench LongCodeQA (Rando et al., 2025): questions about real GitHub repos, bucketed by size.
              Prompt = the dataset's own prompt (goal + repository + question).

Data are fetched from the Hugging Face Hub into $SCRATCH/sota_data (or --data-dir) on first use.
"""
import json
import os
import zipfile
from pathlib import Path

DATA_DIR = Path(os.environ.get("SCRATCH", "/vol")) / "sota_data"
LETTERS = "ABCD"

SOURCES = {
    "lbv2": ("zai-org/LongBench-v2", "data.json"),
    "codedebug": ("xinrongzhang2022/InfiniteBench", "code_debug.jsonl"),
    "lcqa": ("Steefano/LCB", "LongCodeQA.zip"),
}
TASKS = ("lbv2", "codedebug", "lcqa32k", "lcqa64k", "lcqa128k")


def fetch(source, data_dir=DATA_DIR):
    from huggingface_hub import hf_hub_download
    repo, name = SOURCES[source]
    local = Path(data_dir) / repo.split("/")[1]
    path = local / name
    if not path.exists():
        hf_hub_download(repo, name, repo_type="dataset", local_dir=local)
    if name.endswith(".zip") and not (local / "LQA").exists():
        with zipfile.ZipFile(path) as z:
            z.extractall(local)
    return path if not name.endswith(".zip") else local / "LQA"


def _lbv2(data_dir):
    rows = []
    for x in json.loads(fetch("lbv2", data_dir).read_text()):
        rows.append({
            "id": x["_id"], "group": x["domain"], "difficulty": x["difficulty"], "length_bucket": x["length"],
            "prefix": "Please read the following text and answer the question below.\n\n<text>\n",
            "context": x["context"],
            "suffix": (f"\n</text>\n\nWhat is the correct answer to this question: {x['question'].strip()}\n"
                       f"Choices:\n(A) {x['choice_A'].strip()}\n(B) {x['choice_B'].strip()}\n"
                       f"(C) {x['choice_C'].strip()}\n(D) {x['choice_D'].strip()}\n\n"
                       'Format your response as follows: "The correct answer is (insert answer here)".'),
            "gold": x["answer"],
        })
    return rows


def _codedebug(data_dir):
    rows = []
    with open(fetch("codedebug", data_dir)) as f:
        for line in f:
            x = json.loads(line)
            opts = x["options"]
            rows.append({
                "id": str(x["id"]), "group": "Code.Debug",
                "prefix": ("There is ONLY ONE function in the large project that is deliberately made to include an "
                           "obvious error. Please find the function that contains the most obvious errors. I will give "
                           "you four options to narrow your scope. You can inspect the options and think. Eventually, "
                           "tell me the answer using one single letter (A, B, C, or D).\n\n"),
                "context": x["context"],
                "suffix": ("\n\nWhich funtion has deliberate error?\n"
                           + "".join(f"{L}. {o}\n" for L, o in zip(LETTERS, opts))
                           + "\nYou should first find the functions in the options. Repeat their content, inspect "
                             "through code, and at last give me your answer for the function that has the deliberate "
                             "and obvious error in A, B, C, or D."),
                "gold": LETTERS[opts.index(x["answer"][0])],
            })
    return rows


def _lcqa(data_dir, bucket):
    rows = []
    for i, x in enumerate(json.loads((fetch("lcqa", data_dir) / f"{bucket}.json").read_text())):
        prompt, repo = x["prompt"], x["repo_text"]
        cut = prompt.find(repo)
        if cut < 0 or prompt.find(repo, cut + 1) >= 0:
            raise ValueError(f"unexpected LongCodeQA prompt layout at {bucket}[{i}]")
        rows.append({"id": f"{bucket}-{i}", "group": f"LongCodeQA {bucket}", "repo": x["repo"],
                     "prefix": prompt[:cut], "context": repo, "suffix": prompt[cut + len(repo):].rstrip() + "\n",
                     "gold": x["correct_letter"].strip()})
    return rows


def load_task(task, data_dir=DATA_DIR):
    if task == "lbv2":
        rows = _lbv2(data_dir)
    elif task == "codedebug":
        rows = _codedebug(data_dir)
    elif task.startswith("lcqa"):
        rows = _lcqa(data_dir, task[4:].upper())
    else:
        raise ValueError(f"unknown task {task}; choose from {TASKS}")
    for r in rows:
        if r["gold"] not in LETTERS:
            raise ValueError(f"{task}/{r['id']}: gold {r['gold']!r}")
        r["task"] = task
    return rows
