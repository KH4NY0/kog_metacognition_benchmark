"""
task1_preanswerpredict.py — KnowOrGuess Task 1
Prospective metacognitive monitoring: can the model predict its own accuracy
before seeing a question, based only on domain and difficulty metadata?
"""

import kaggle_benchmarks as kb
from utils import (
    load_mmlu_sample,
    build_prediction_prompt,
    build_qa_prompt,
    parse_integer,
    parse_answer_letter,
    brier_score,
)


# ── Task Definition ───────────────────────────────────────────────────────────

class PreAnswerPredictTask(kb.Task):
    """
    Two-turn task:
      Turn 1 — model predicts P(correct) given only domain + difficulty
      Turn 2 — model answers the actual question
    Score = Brier Score on predictions vs actual outcomes (lower = better)
    """

    name = "pre_answer_predict"
    description = (
        "Prospective metacognitive monitoring. "
        "The model predicts its own accuracy before seeing each question."
    )

    def load_data(self) -> list[dict]:
        return load_mmlu_sample(n_per_difficulty=50, seed=42)

    def build_turns(self, record: dict) -> list[kb.Turn]:
        """
        Returns two turns per record:
          - Turn A: prediction prompt (metadata only)
          - Turn B: full question for actual answering
        """
        turn_a = kb.Turn(
            turn_id=f"{record['question_id']}_predict",
            prompt=build_prediction_prompt(record["domain"], record["difficulty"]),
            metadata={
                "question_id": record["question_id"],
                "phase": "predict",
                "domain": record["domain"],
                "difficulty": record["difficulty"],
            },
        )
        turn_b = kb.Turn(
            turn_id=f"{record['question_id']}_answer",
            prompt=build_qa_prompt(
                record["question"], record["choices"], ask_confidence=False
            ),
            metadata={
                "question_id": record["question_id"],
                "phase": "answer",
                "correct_answer": record["correct_answer"],
            },
        )
        return [turn_a, turn_b]

    def score(self, responses: list[kb.Response]) -> kb.ScoreResult:
        """
        Pair prediction and answer responses by question_id, compute Brier Score.
        """
        predictions_map: dict[str, float] = {}
        actuals_map: dict[str, int] = {}

        for resp in responses:
            qid = resp.metadata["question_id"]
            phase = resp.metadata["phase"]

            if phase == "predict":
                raw = parse_integer(resp.text, lo=0, hi=100)
                predictions_map[qid] = (raw / 100.0) if raw is not None else 0.5

            elif phase == "answer":
                letter = parse_answer_letter(resp.text)
                correct = resp.metadata["correct_answer"]
                actuals_map[qid] = 1 if letter == correct else 0

        # Only score questions where both phases are present
        shared_ids = sorted(predictions_map.keys() & actuals_map.keys())
        if not shared_ids:
            return kb.ScoreResult(score=1.0, details={"error": "No paired responses"})

        preds = [predictions_map[qid] for qid in shared_ids]
        acts = [actuals_map[qid] for qid in shared_ids]
        bs = brier_score(preds, acts)

        # Per-difficulty breakdown
        breakdown = {}
        difficulties = set()
        for resp in responses:
            if resp.metadata["phase"] == "predict":
                difficulties.add(resp.metadata["difficulty"])

        for diff in difficulties:
            diff_ids = [
                qid for qid in shared_ids
                if any(
                    r.metadata.get("difficulty") == diff
                    and r.metadata.get("question_id") == qid
                    for r in responses
                )
            ]
            if diff_ids:
                dp = [predictions_map[qid] for qid in diff_ids]
                da = [actuals_map[qid] for qid in diff_ids]
                breakdown[diff] = round(brier_score(dp, da), 4)

        return kb.ScoreResult(
            score=round(1.0 - bs, 4),           # invert so higher = better
            details={
                "brier_score": round(bs, 4),
                "n_questions": len(shared_ids),
                "parse_failures": len(shared_ids) - sum(
                    1 for p in preds if p != 0.5
                ),
                "per_difficulty": breakdown,
            },
        )


# ── Entry Point ───────────────────────────────────────────────────────────────

def get_task() -> kb.Task:
    return PreAnswerPredictTask()