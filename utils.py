"""
utils.py — Shared utilities for the KnowOrGuess benchmark.
Handles MMLU data loading, prompt construction, and scoring helpers.
"""

import re
import math
from datasets import load_dataset

# ── Constants ────────────────────────────────────────────────────────────────

DIFFICULTY_SUBJECTS = {
    "easy": [
        "high_school_geography",
        "high_school_us_history",
        "elementary_mathematics",
    ],
    "medium": [
        "high_school_biology",
        "high_school_chemistry",
        "high_school_psychology",
    ],
    "hard": [
        "college_mathematics",
        "college_physics",
        "college_computer_science",
    ],
    "expert": [
        "professional_medicine",
        "professional_law",
        "abstract_algebra",
    ],
}

ANSWER_LABELS = ["A", "B", "C", "D"]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_mmlu_sample(n_per_difficulty: int = 50, seed: int = 42) -> list[dict]:
    """
    Load a balanced MMLU sample across four difficulty tiers.

    Returns a flat list of records, each with:
        question_id, domain, difficulty, question, choices, correct_answer
    """
    records = []
    record_id = 0

    for difficulty, subjects in DIFFICULTY_SUBJECTS.items():
        per_subject = max(1, n_per_difficulty // len(subjects))

        for subject in subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
            except Exception as e:
                print(f"[warn] Could not load {subject}: {e}")
                continue

            ds = ds.shuffle(seed=seed).select(range(min(per_subject, len(ds))))

            for row in ds:
                choices = [row["choices"][i] for i in range(4)]
                correct_label = ANSWER_LABELS[row["answer"]]

                records.append({
                    "question_id": f"q{record_id:04d}",
                    "domain": subject.replace("_", " ").title(),
                    "difficulty": difficulty,
                    "question": row["question"],
                    "choices": choices,
                    "correct_answer": correct_label,
                })
                record_id += 1

    return records


# ── Prompt Templates ──────────────────────────────────────────────────────────

def build_prediction_prompt(domain: str, difficulty: str) -> str:
    """Task 1 — shown only metadata, not the question."""
    return (
        f"You are about to answer a multiple-choice question.\n"
        f"Domain: {domain}\n"
        f"Difficulty: {difficulty.title()}\n\n"
        f"Before you see the question, estimate the probability (0–100) "
        f"that you will answer it correctly.\n\n"
        f"Reply with ONLY a single integer between 0 and 100. No explanation."
    )


def build_qa_prompt(question: str, choices: list[str], ask_confidence: bool = True) -> str:
    """Task 2 — standard QA with optional confidence elicitation."""
    choices_text = "\n".join(
        f"{label}. {text}" for label, text in zip(ANSWER_LABELS, choices)
    )
    base = (
        f"Answer the following multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"{choices_text}\n\n"
    )
    if ask_confidence:
        return base + (
            "Reply in this exact format:\n"
            "Answer: <letter>\n"
            "Confidence: <integer 0-100>"
        )
    return base + "Reply with ONLY the answer letter (A, B, C, or D)."


def build_audit_prompt(qa_pairs: list[dict]) -> str:
    """
    Task 3 — give model a batch of its own Q&A pairs, ask it to flag errors.

    Each item in qa_pairs should have: question, choices, model_answer
    """
    lines = ["Review the following questions and the answers that were given.\n"]
    for i, item in enumerate(qa_pairs, 1):
        choices_text = " | ".join(
            f"{label}. {text}"
            for label, text in zip(ANSWER_LABELS, item["choices"])
        )
        lines.append(
            f"[{i}] Question: {item['question']}\n"
            f"     Options: {choices_text}\n"
            f"     Answer given: {item['model_answer']}\n"
        )

    lines.append(
        "\nIdentify which answers above are INCORRECT.\n"
        "Reply with ONLY a comma-separated list of the item numbers you believe "
        "are wrong (e.g. '2, 5, 7'). If you believe all are correct, reply '0'."
    )
    return "\n".join(lines)


# ── Output Parsing ────────────────────────────────────────────────────────────

def parse_integer(text: str, lo: int = 0, hi: int = 100) -> int | None:
    """Extract the first integer in [lo, hi] from model output."""
    matches = re.findall(r"\b(\d{1,3})\b", text)
    for m in matches:
        val = int(m)
        if lo <= val <= hi:
            return val
    return None


def parse_answer_letter(text: str) -> str | None:
    """Extract answer letter A/B/C/D from model output."""
    match = re.search(r"\bAnswer\s*:\s*([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: first standalone letter
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1).upper() if match else None


def parse_confidence(text: str) -> int | None:
    """Extract confidence score from Task 2 output."""
    match = re.search(r"Confidence\s*:\s*(\d{1,3})", text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        return max(0, min(100, val))
    return parse_integer(text)


def parse_flagged_items(text: str, n_items: int = 10) -> set[int]:
    """Extract flagged item numbers from Task 3 output."""
    if text.strip() == "0":
        return set()
    nums = re.findall(r"\b(\d+)\b", text)
    return {int(n) for n in nums if 1 <= int(n) <= n_items}


# ── Scoring Functions ─────────────────────────────────────────────────────────

def brier_score(predictions: list[float], actuals: list[int]) -> float:
    """
    Brier Score for Task 1.
    predictions: list of probabilities in [0, 1]
    actuals:     list of 0/1 (0 = wrong, 1 = correct)
    Lower is better. Perfect = 0.0, Worst = 1.0
    """
    if not predictions:
        return 1.0
    return sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)


def expected_calibration_error(
    confidences: list[float],
    correct: list[int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) for Task 2.
    confidences: list of floats in [0, 1]
    correct:     list of 0/1
    Lower is better.
    """
    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(confidences)

    for b in range(n_bins):
        lo, hi = b * bin_size, (b + 1) * bin_size
        indices = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not indices:
            continue
        avg_conf = sum(confidences[i] for i in indices) / len(indices)
        avg_acc = sum(correct[i] for i in indices) / len(indices)
        ece += (len(indices) / n) * abs(avg_conf - avg_acc)

    return ece


def error_detection_f1(
    flagged: set[int],
    actual_errors: set[int],
) -> dict:
    """
    F1 score for Task 3.
    flagged:       item numbers the model flagged as wrong
    actual_errors: item numbers that are actually wrong
    """
    tp = len(flagged & actual_errors)
    fp = len(flagged - actual_errors)
    fn = len(actual_errors - flagged)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}