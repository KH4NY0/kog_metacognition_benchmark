# KnowOrGuess: Metacognition Benchmark

![KnowOrGuess Banner](Know%20Or%20Guess%20Banner.png)

A three-stage benchmark for evaluating metacognitive ability in frontier language models,
submitted to the Google DeepMind × Kaggle "Measuring Progress Toward AGI" hackathon.

## Project Structure

```
knoworguess/
├── utils.py                    # Shared: data loading, prompts, scoring helpers
├── task1_preanswerpredict.py   # Task 1: Prospective monitoring (Brier Score)
├── task2_confidencecalibrate.py # Task 2: Confidence calibration (ECE)
├── task3_selferrordetect.py    # Task 3: Retrospective error detection (F1)
├── benchmark.py                # Benchmark assembler + CLI entrypoint
└── README.md
```

## Setup

```bash
pip install kaggle-benchmarks datasets
```

## Running

```bash
# Dry run (prints task info, no model calls)
python benchmark.py

# Run full benchmark via Kaggle CLI
kaggle benchmarks run benchmark.py --model gpt-4o
kaggle benchmarks run benchmark.py --model claude-sonnet-4-5
kaggle benchmarks run benchmark.py --model gemini-1.5-pro
```

## Scoring

| Task | Metric | Weight |
|---|---|---|
| Task 1: PreAnswerPredict | 1 − Brier Score | 30% |
| Task 2: ConfidenceCalibrate | 1 − ECE | 35% |
| Task 3: SelfErrorDetect | F1 | 35% |

All scores in [0, 1]. Higher = better.

## Dataset

MMLU (Hendrycks et al., 2021) — 200 questions, 50 per difficulty tier
(Easy / Medium / Hard / Expert), balanced across STEM, Humanities,
Social Sciences, and Other domains.

Source: `cais/mmlu` on HuggingFace. MIT License.

## Track

Metacognition: Google DeepMind × Kaggle AGI Hackathon 2026.