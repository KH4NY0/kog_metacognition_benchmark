"""
benchmark.py — KnowOrGuess Benchmark
Assembles the three metacognitive tasks and defines the composite benchmark.

Usage:
    python benchmark.py                 # dry run, prints task info
    kaggle benchmarks run benchmark.py  # execute via kaggle-benchmarks CLI
"""

import kaggle_benchmarks as kb

from task1_preanswerpredict import get_task as get_task1
from task2_confidencecalibrate import get_task as get_task2
from task3_selferrordetect import get_task as get_task3


# ── Benchmark Definition ──────────────────────────────────────────────────────

benchmark = kb.Benchmark(
    name="KnowOrGuess",
    description=(
        "A three-stage metacognitive benchmark for frontier language models. "
        "Isolates prospective monitoring (Task 1: PreAnswerPredict), "
        "concurrent confidence calibration (Task 2: ConfidenceCalibrate), "
        "and retrospective error detection (Task 3: SelfErrorDetect). "
        "Based on MMLU questions across four difficulty tiers."
    ),
    tasks=[
        get_task1(),
        get_task2(),
        get_task3(),
    ],
    scoring=kb.CompositeScoring(
        weights={
            "pre_answer_predict": 0.30,
            "confidence_calibrate": 0.35,
            "self_error_detect": 0.35,
        },
        aggregation="weighted_mean",
    ),
    tags=["metacognition", "calibration", "self-monitoring", "agi-evaluation"],
)


# ── Leaderboard Column Definitions ───────────────────────────────────────────

benchmark.add_leaderboard_columns([
    kb.Column("composite_score",         label="Overall",          primary=True),
    kb.Column("pre_answer_predict",      label="Task 1: Predict"),
    kb.Column("confidence_calibrate",    label="Task 2: Calibrate"),
    kb.Column("self_error_detect",       label="Task 3: Audit"),
])


# ── Dry-Run Entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Benchmark : {benchmark.name}")
    print(f"  Tasks     : {len(benchmark.tasks)}")
    for task in benchmark.tasks:
        print(f"    - {task.name}: {task.description[:60]}...")
    print(f"{'='*60}\n")

    print("Scoring weights:")
    for task_name, weight in benchmark.scoring.weights.items():
        print(f"  {task_name}: {weight:.0%}")

    print("\nTo run against a model:")
    print("  kaggle benchmarks run benchmark.py --model gpt-4o")
    print("  kaggle benchmarks run benchmark.py --model claude-sonnet-4-5")
    print("  kaggle benchmarks run benchmark.py --model gemini-1.5-pro\n")