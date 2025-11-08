#!/usr/bin/env python3
"""Convenience wrapper to run the standard SPRT scheduling experiments."""

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
CARGO_BIN = ["cargo", "run", "--release", "--"]

# (stopping, policy, label)
EXPERIMENTS: List[Tuple[str, str, str]] = [
    ("complete", "sequential", "complete sequential"),
    ("ucb-dominance", "sequential", "ucb-dominance seq"),
    ("ucb-dominance", "ucb", "ucb-dominance ucb"),
    ("ucb-dominance", "thompson", "ucb-dominance thompson"),
    ("stop-at-first", "sequential", "first seq"),
    ("stop-at-first", "thompson", "first thompson"),
    ("stop-at-first", "ucb", "first ucb"),
]


def append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def run_experiment(runs: int, stopping: str, policy: str) -> str:
    """Invoke the Rust simulator and return combined stdout/stderr."""

    cmd = [
        *CARGO_BIN,
        "--runs",
        str(runs),
        "--stopping",
        stopping,
        "--policy",
        policy,
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical SPRT scheduling experiments and log the results.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Number of Monte Carlo repetitions to use for each configuration (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_runs.log"),
        help="File to append simulator outputs to (default: experiment_runs.log).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    header = f"\n\n===== Experiment batch @ {timestamp} (runs={args.runs}) =====\n"
    args.output.write_text("", encoding="utf-8")  # truncate existing contents
    append_text(args.output, header)

    for stopping, policy, label in EXPERIMENTS:
        print(f"→ Running {label}: stopping={stopping}, policy={policy}")
        append_text(
            args.output,
            f"\n--- {label} (stopping={stopping}, policy={policy}) ---\n",
        )
        output = run_experiment(args.runs, stopping, policy)
        print(output)
        append_text(args.output, output)


if __name__ == "__main__":
    main()
