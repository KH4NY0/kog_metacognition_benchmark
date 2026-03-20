"""
task2_confidencecalibrate.py — KnowOrGuess Task 2
Concurrent confidence monitoring: does the model's stated confidence
accurately reflect its actual accuracy? Measured via Expected Calibration Error.
"""

import kaggle_benchmarks as kb
from utils import (
    load_mmlu_sample,
    build_qa_prompt,
    parse_answer_letter,
    parse_confidence,
    expected_calibration_error,
)


# ── Task Definition ───────────────────────────────────────────────────────────

class ConfidenceCalibrateTask(kb.Task):
    """
    Single-turn task: model answers a multiple-choice question and states
    a confidence score (0–100) in the same response.

    Score = 1 - ECE (higher = better calibrated)

    A well-calibrated model that says "70% confident" should be correct
    approximately 70% of the time across all questions where it said that.
    """

    name = "confidence_calibrate"
    description = (
        "Concurrent confidence monitoring. "
        "Measures whether stated confidence correlates with actual accuracy "
        "using Expected Calibration Error (ECE)."
    )

    def load_data(self) -> list[dict]:
        return load_mmlu_sample(n_per_difficulty=50, seed=42)

    def build_turns(self, record: dict) -> list[kb.Turn]:
        return [
            kb.Turn(
                turn_id=record["question_id"],
                prompt=build_qa_prompt(
                    record["question"],
                    record["choices"],
                    ask_confidence=True,
                ),
                metadata={
                    "question_id": record["question_id"],
                    "domain": record["domain"],
                    "difficulty": record["difficulty"],
                    "correct_answer": record["correct_answer"],
                },
            )
        ]

    def score(self, responses: list[kb.Response]) -> kb.ScoreResult:
        confidences: list[float] = []
        correct_flags: list[int] = []
        parse_failures = 0
        per_difficulty: dict[str, list] = {}

        for resp in responses:
            letter = parse_answer_letter(resp.text)
            conf_raw = parse_confidence(resp.text)

            if letter is None or conf_raw is None:
                parse_failures += 1
                continue

            is_correct = int(letter == resp.metadata["correct_answer"])
            conf_norm = conf_raw / 100.0

            confidences.append(conf_norm)
            correct_flags.append(is_correct)

            diff = resp.metadata["difficulty"]
            if diff not in per_difficulty:
                per_difficulty[diff] = {"confs": [], "corrects": []}
            per_difficulty[diff]["confs"].append(conf_norm)
            per_difficulty[diff]["corrects"].append(is_correct)

        if not confidences:
            return kb.ScoreResult(
                score=0.0,
                details={"error": "All responses failed to parse"},
            )

        ece = expected_calibration_error(confidences, correct_flags, n_bins=10)
        overall_accuracy = sum(correct_flags) / len(correct_flags)
        avg_confidence = sum(confidences) / len(confidences)
        overconfidence_gap = avg_confidence - overall_accuracy

        breakdown = {}
        for diff, data in per_difficulty.items():
            diff_ece = expected_calibration_error(
                data["confs"], data["corrects"], n_bins=5
            )
            breakdown[diff] = {
                "ece": round(diff_ece, 4),
                "n": len(data["confs"]),
                "accuracy": round(sum(data["corrects"]) / len(data["corrects"]), 4),
                "avg_confidence": round(
                    sum(data["confs"]) / len(data["confs"]), 4
                ),
            }

        return kb.ScoreResult(
            score=round(1.0 - ece, 4),   # invert so higher = better
            details={
                "ece": round(ece, 4),
                "overall_accuracy": round(overall_accuracy, 4),
                "avg_confidence": round(avg_confidence, 4),
                "overconfidence_gap": round(overconfidence_gap, 4),
                "n_questions": len(confidences),
                "parse_failures": parse_failures,
                "per_difficulty": breakdown,
            },
        )


# ── Entry Point ───────────────────────────────────────────────────────────────

def get_task() -> kb.Task:
    return ConfidenceCalibrateTask()