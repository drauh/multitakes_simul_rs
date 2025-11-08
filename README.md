# Multi-Takes Simulation (multitakes_simul_rs)

This repository contains a Monte Carlo simulation and supporting code used to study strategies for managing "multi-take" Fishtest runs (groups of related chess engine tests). The simulations compare allocation policies and stopping rules to find an efficient, statistically sound scheduler for grouping related tests.

Summary
-------

- The simulation models seven candidate patches (corradj1..corradj7) with known true Elo strengths. The goal is to select the strongest candidate while minimizing total games played across the group.
- Several allocation policies (Sequential, UCB, Thompson Sampling) and stopping conditions (Complete, Stop-at-First, UCB-Dominance) were compared over many runs.

Key artifacts and code
----------------------

- `src/` - Rust implementation of the simulation and statistical utilities (see `main.rs`, `sprt.rs`, `simulation.rs`, etc.).
- `run_experiments.py` - helper script to run and aggregate experiments (see the script for usage details).
- `artifacts/10000_runs.txt` - output from a batch of simulation runs.
