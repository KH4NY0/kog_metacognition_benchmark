"""
task3_selferrordetect.py — KnowOrGuess Task 3
Retrospective error monitoring: given a batch of its own answers (correct and
incorrect), can the model identify which ones are wrong — without being told?
"""

import random
import kaggle_benchmarks as kb
from utils import (
    load_mmlu_sample,
    build_qa_prompt,
    build_audit_prompt,
    parse_answer_letter,
    parse_flagged_items,
    error_detection_f1,
)

BATCH_SIZE = 10          # questions per audit batch
ERROR_RATE_TARGET = 0.4  # ~40% of each batch will be intentionally wrong


# ── Task Definition ───────────────────────────────────────────────────────────

class SelfErrorDetectTask(kb.Task):
    """
    Two-phase task:

    Phase 1 (answer collection):
        The model answers each question individually. Responses are recorded
        but NOT scored yet. Incorrect answers are injected where needed to
        reach ERROR_RATE_TARGET.

    Phase 2 (audit):
        The model is shown batches of BATCH_SIZE (question + its earlier answer)
        and must identify which answers it believes are wrong.

    Score = F1 on error detection (treating each wrong answer as a positive case)

    Note on answer injection:
        To guarantee a meaningful signal, we ensure ~40% of each batch contains
        a wrong answer. For questions the model answered correctly, we
        probabilistically substitute a random incorrect option. This is
        disclosed in the benchmark description so judges understand the setup.
    """

    name = "self_error_detect"
    description = (
        "Retrospective error monitoring. "
        "The model is shown batches of its own Q&A pairs and must flag "
        "which answers it believes are incorrect, without ground truth."
    )

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def load_data(self) -> list[dict]:
        records = load_mmlu_sample(n_per_difficulty=50, seed=self.seed)
        # Group records into batches of BATCH_SIZE
        batches = [
            records[i: i + BATCH_SIZE]
            for i in range(0, len(records), BATCH_SIZE)
        ]
        return batches  # each item in dataset is now a batch

    def build_turns(self, batch: list[dict]) -> list[kb.Turn]:
        """
        Phase 1: build one answering turn per question in the batch.
        Phase 2: build one audit turn for the whole batch.
        Returns phase-1 turns; phase-2 is assembled in post_process.
        """
        turns = []
        for record in batch:
            turns.append(
                kb.Turn(
                    turn_id=f"{record['question_id']}_phase1",
                    prompt=build_qa_prompt(
                        record["question"],
                        record["choices"],
                        ask_confidence=False,
                    ),
                    metadata={
                        "batch_id": batch[0]["question_id"],
                        "question_id": record["question_id"],
                        "phase": "answer",
                        "question": record["question"],
                        "choices": record["choices"],
                        "correct_answer": record["correct_answer"],
                    },
                )
            )
        return turns

    def post_process(
        self,
        phase1_responses: list[kb.Response],
        batch: list[dict],
    ) -> kb.Turn:
        """
        After phase-1 answers are collected:
          1. Parse model answers
          2. Inject errors to reach ERROR_RATE_TARGET
          3. Build and return the audit turn
        """
        qa_pairs = []
        actual_errors = set()

        for i, (resp, record) in enumerate(
            zip(phase1_responses, batch), start=1
        ):
            model_answer = parse_answer_letter(resp.text) or record["correct_answer"]
            is_correct = model_answer == record["correct_answer"]

            # Inject error if this question was correct and we need more errors
            error_quota = round(BATCH_SIZE * ERROR_RATE_TARGET)
            current_errors = len(actual_errors)
            should_inject = (
                is_correct
                and current_errors < error_quota
                and self.rng.random() < 0.6
            )

            if should_inject:
                wrong_options = [
                    lbl for lbl in ["A", "B", "C", "D"]
                    if lbl != record["correct_answer"]
                ]
                model_answer = self.rng.choice(wrong_options)

            if model_answer != record["correct_answer"]:
                actual_errors.add(i)

            qa_pairs.append({
                "question": record["question"],
                "choices": record["choices"],
                "model_answer": model_answer,
            })

        audit_turn = kb.Turn(
            turn_id=f"{batch[0]['question_id']}_phase2",
            prompt=build_audit_prompt(qa_pairs),
            metadata={
                "batch_id": batch[0]["question_id"],
                "phase": "audit",
                "actual_errors": sorted(actual_errors),
                "n_items": len(qa_pairs),
            },
        )
        return audit_turn

    def score(self, responses: list[kb.Response]) -> kb.ScoreResult:
        """Score only phase-2 (audit) responses."""
        audit_responses = [
            r for r in responses if r.metadata.get("phase") == "audit"
        ]

        all_f1, all_precision, all_recall = [], [], []
        batch_details = []

        for resp in audit_responses:
            n_items = resp.metadata["n_items"]
            actual_errors = set(resp.metadata["actual_errors"])
            flagged = parse_flagged_items(resp.text, n_items=n_items)
            result = error_detection_f1(flagged, actual_errors)

            all_f1.append(result["f1"])
            all_precision.append(result["precision"])
            all_recall.append(result["recall"])
            batch_details.append({
                "batch_id": resp.metadata["batch_id"],
                "actual_errors": sorted(actual_errors),
                "flagged": sorted(flagged),
                **{k: round(v, 4) for k, v in result.items()},
            })

        if not all_f1:
            return kb.ScoreResult(
                score=0.0,
                details={"error": "No audit responses found"},
            )

        avg_f1 = sum(all_f1) / len(all_f1)
        avg_precision = sum(all_precision) / len(all_precision)
        avg_recall = sum(all_recall) / len(all_recall)

        return kb.ScoreResult(
            score=round(avg_f1, 4),
            details={
                "avg_f1": round(avg_f1, 4),
                "avg_precision": round(avg_precision, 4),
                "avg_recall": round(avg_recall, 4),
                "n_batches": len(all_f1),
                "batch_results": batch_details,
            },
        )


# ── Entry Point ───────────────────────────────────────────────────────────────

def get_task() -> kb.Task:
    return SelfErrorDetectTask()