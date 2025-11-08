use std::cmp::Ordering;
use std::fmt;

use anyhow::Result;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;

use crate::pentanomial::pentanomial_true_elo;
use crate::sprt::{EloModel, SprtParams, sprt_elo, sprt_quantile};

const TASK_SIZE: usize = 96;
pub const DEFAULT_INITIAL_TASKS: usize = 20;
const LLR_TARGET: f64 = 2.94;
const BUDGET_CAP: usize = 10_000_000;

pub const SPRT_PARAMS: SprtParams = SprtParams {
    alpha: 0.05,
    beta: 0.05,
    elo0: 0.0,
    elo1: 2.0,
    elo_model: EloModel::Normalized,
    p: 0.05,
};

#[derive(Clone, Copy)]
struct CandidateSeed {
    name: &'static str,
    _games: usize,
    penta: [u64; 5],
}

const CANDIDATE_DATA: [CandidateSeed; 7] = [
    CandidateSeed {
        name: "corradj1",
        _games: 142_688,
        penta: [461, 16_726, 36_723, 16_920, 514],
    },
    CandidateSeed {
        name: "corradj2",
        _games: 21_856,
        penta: [82, 2_646, 5_623, 2_507, 70],
    },
    CandidateSeed {
        name: "corradj3",
        _games: 102_464,
        penta: [346, 12_061, 26_295, 12_166, 364],
    },
    CandidateSeed {
        name: "corradj4",
        _games: 45_984,
        penta: [159, 5_353, 11_766, 5_551, 163],
    },
    CandidateSeed {
        name: "corradj5",
        _games: 15_968,
        penta: [58, 1_922, 4_002, 1_945, 57],
    },
    CandidateSeed {
        name: "corradj6",
        _games: 20_288,
        penta: [77, 2_403, 5_217, 2_388, 59],
    },
    CandidateSeed {
        name: "corradj7",
        _games: 13_824,
        penta: [43, 1_633, 3_539, 1_658, 39],
    },
];

const NUM_CANDIDATES: usize = CANDIDATE_DATA.len();

struct Candidate {
    seed: CandidateSeed,
    penta_prob: Vec<f64>,
    sampler: WeightedIndex<f64>,
    current_penta: [u64; 5],
    current_games: usize,
    pub elo: f64,
    pub lcb: f64,
    pub ucb: f64,
    pub llr: f64,
    pub is_active: bool,
}

impl Clone for Candidate {
    fn clone(&self) -> Self {
        let sampler = WeightedIndex::new(self.penta_prob.clone()).expect("valid distribution");
        Self {
            seed: self.seed,
            penta_prob: self.penta_prob.clone(),
            sampler,
            current_penta: self.current_penta,
            current_games: self.current_games,
            elo: self.elo,
            lcb: self.lcb,
            ucb: self.ucb,
            llr: self.llr,
            is_active: self.is_active,
        }
    }
}

impl Candidate {
    fn new(seed: CandidateSeed) -> Self {
        let total: u64 = seed.penta.iter().sum();
        let probs: Vec<f64> = if total == 0 {
            vec![0.2; 5]
        } else {
            seed.penta
                .iter()
                .map(|&c| c as f64 / total as f64)
                .collect()
        };
        let sampler = WeightedIndex::new(probs.clone()).expect("valid distribution");
        Self {
            seed,
            penta_prob: probs,
            sampler,
            current_penta: [0; 5],
            current_games: 0,
            elo: 0.0,
            lcb: f64::NEG_INFINITY,
            ucb: f64::INFINITY,
            llr: 0.0,
            is_active: true,
        }
    }

    fn add_games<R: Rng + ?Sized>(&mut self, games: usize, rng: &mut R) {
        if games == 0 {
            return;
        }
        let pairs = games / 2;
        if pairs == 0 {
            return;
        }
        for _ in 0..pairs {
            let idx = self.sampler.sample(rng);
            self.current_penta[idx] += 1;
        }
        self.current_games += pairs * 2;
    }

    fn update_stats(&mut self, params: &SprtParams) -> Result<()> {
        if self.current_games == 0 {
            return Ok(());
        }
        let stats = sprt_elo(&self.current_penta, params)?;
        self.llr = stats.llr;
        self.elo = stats.elo;
        self.lcb = stats.ci.0;
        self.ucb = stats.ci.1;
        Ok(())
    }

    fn ucb_score(&self) -> f64 {
        if self.ucb.is_nan() {
            f64::NEG_INFINITY
        } else {
            self.ucb
        }
    }

    fn thompson_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<f64> {
        if self.current_games == 0 {
            return Ok(f64::NEG_INFINITY);
        }
        let percentile = Uniform::new(1e-9, 1.0 - 1e-9).sample(rng);
        sprt_quantile(&self.current_penta, &SPRT_PARAMS, percentile)
    }
}

impl fmt::Display for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Candidate(name='{}', games={}, LLR={:.2}, LCB={:.2}, UCB={:.2}, active={})",
            self.seed.name, self.current_games, self.llr, self.lcb, self.ucb, self.is_active
        )
    }
}

#[derive(Clone, Copy)]
pub struct StatsBlock {
    pub llr: f64,
    pub lcb: f64,
    pub ucb: f64,
    pub elo: f64,
    pub games: f64,
}

impl Default for StatsBlock {
    fn default() -> Self {
        zero_block()
    }
}

fn zero_block() -> StatsBlock {
    StatsBlock {
        llr: 0.0,
        lcb: 0.0,
        ucb: 0.0,
        elo: 0.0,
        games: 0.0,
    }
}

impl StatsBlock {
    fn accumulate(&mut self, other: &StatsBlock) {
        self.llr += other.llr;
        self.lcb += other.lcb;
        self.ucb += other.ucb;
        self.elo += other.elo;
        self.games += other.games;
    }

    fn average(&self, runs: f64) -> StatsBlock {
        if runs <= 0.0 {
            return *self;
        }
        StatsBlock {
            llr: self.llr / runs,
            lcb: self.lcb / runs,
            ucb: self.ucb / runs,
            elo: self.elo / runs,
            games: self.games / runs,
        }
    }

    fn is_populated(&self) -> bool {
        self.games > 0.0
    }
}

fn stats_block_from_candidate(candidate: &Candidate) -> StatsBlock {
    StatsBlock {
        llr: candidate.llr,
        lcb: candidate.lcb,
        ucb: candidate.ucb,
        elo: candidate.elo,
        games: candidate.current_games as f64,
    }
}

#[derive(Clone, Copy)]
pub struct ComparisonRow {
    pub name: &'static str,
    pub average: StatsBlock,
    pub completion: StatsBlock,
    pub true_elo: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationPolicy {
    Ucb,
    ThompsonSampling,
    Sequential,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoppingCondition {
    UcbDominance,
    StopAtFirst,
    Complete,
}

#[derive(Clone, Copy)]
pub struct SimulationResult {
    pub winner_index: Option<usize>,
    pub sim_games: usize,
    pub games_to_first_win: Option<usize>,
    pub average_stats: [StatsBlock; NUM_CANDIDATES],
    pub completion_stats: [StatsBlock; NUM_CANDIDATES],
}

fn run_once_with_seed(
    verbose: bool,
    stopping: StoppingCondition,
    policy: AllocationPolicy,
    initial_tasks: usize,
    seed: u64,
) -> Result<SimulationResult> {
    let mut rng = SmallRng::seed_from_u64(seed);
    run_once_with_rng(verbose, stopping, policy, initial_tasks, &mut rng)
}

fn run_once_with_rng<R: Rng + ?Sized>(
    verbose: bool,
    stopping: StoppingCondition,
    policy: AllocationPolicy,
    initial_tasks: usize,
    rng: &mut R,
) -> Result<SimulationResult> {
    let (winner_index, sim_games, first_win_games, final_state, completion_state) =
        run_simulation(verbose, stopping, policy, initial_tasks, rng)?;
    Ok(SimulationResult {
        winner_index,
        sim_games,
        games_to_first_win: first_win_games,
        average_stats: final_state,
        completion_stats: completion_state,
    })
}

fn run_simulation<R: Rng + ?Sized>(
    verbose: bool,
    stopping: StoppingCondition,
    policy: AllocationPolicy,
    initial_tasks: usize,
    rng: &mut R,
) -> Result<(
    Option<usize>,
    usize,
    Option<usize>,
    [StatsBlock; NUM_CANDIDATES],
    [StatsBlock; NUM_CANDIDATES],
)> {
    let mut candidates: Vec<Candidate> =
        CANDIDATE_DATA.iter().copied().map(Candidate::new).collect();
    let mut total_games = 0usize;
    let mut best_candidate: Option<usize> = None;
    let mut best_elo = f64::NEG_INFINITY;
    let mut completion_snapshots = [StatsBlock::default(); NUM_CANDIDATES];
    let mut first_win_games: Option<usize> = None;
    let initial_games = TASK_SIZE * initial_tasks;

    if verbose {
        println!("--- INITIALIZATION PHASE ---");
    }
    for (idx, cand) in candidates.iter_mut().enumerate() {
        let before = cand.current_games;
        cand.add_games(initial_games, rng);
        cand.update_stats(&SPRT_PARAMS)?;
        total_games += cand.current_games - before;
        if (cand.llr <= -LLR_TARGET || cand.llr >= LLR_TARGET)
            && !completion_snapshots[idx].is_populated()
        {
            completion_snapshots[idx] = stats_block_from_candidate(cand);
        }
    }
    if verbose {
        println!("Total Games: {}", total_games);
        for cand in &candidates {
            println!("{}", cand);
        }
        println!("--------------------------");
    }

    let complete_mode = matches!(stopping, StoppingCondition::Complete);
    let stop_at_first = matches!(stopping, StoppingCondition::StopAtFirst) && !complete_mode;

    let mut iteration = 0usize;
    let mut seq_cursor = 0usize;
    while total_games < BUDGET_CAP {
        iteration += 1;
        let active_indices: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_active)
            .map(|(idx, _)| idx)
            .collect();
        if active_indices.is_empty() {
            if verbose {
                println!("All candidates pruned. No winner found.");
            }
            break;
        }
        let next_index = match policy {
            AllocationPolicy::Ucb => active_indices
                .into_iter()
                .max_by(|a, b| {
                    candidates[*a]
                        .ucb_score()
                        .partial_cmp(&candidates[*b].ucb_score())
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap(),
            AllocationPolicy::ThompsonSampling => {
                let mut best_idx = None;
                let mut best_sample = f64::NEG_INFINITY;
                for idx in active_indices {
                    let sample = candidates[idx].thompson_sample(rng)?;
                    if sample > best_sample {
                        best_sample = sample;
                        best_idx = Some(idx);
                    }
                }
                best_idx.unwrap()
            }
            AllocationPolicy::Sequential => {
                let mut attempts = 0usize;
                let mut candidate_idx = seq_cursor % NUM_CANDIDATES;
                let selected = loop {
                    if candidates[candidate_idx].is_active {
                        break Some(candidate_idx);
                    }
                    attempts += 1;
                    if attempts >= NUM_CANDIDATES {
                        break None;
                    }
                    candidate_idx = (candidate_idx + 1) % NUM_CANDIDATES;
                };
                if let Some(idx) = selected {
                    seq_cursor = (idx + 1) % NUM_CANDIDATES;
                    idx
                } else {
                    // fallback to first active index (shouldn't happen because we exit when none active)
                    active_indices[0]
                }
            }
        };
        let cand = &mut candidates[next_index];
        let before = cand.current_games;
        cand.add_games(TASK_SIZE, rng);
        total_games += cand.current_games - before;
        cand.update_stats(&SPRT_PARAMS)?;

        if (cand.llr <= -LLR_TARGET || cand.llr >= LLR_TARGET)
            && !completion_snapshots[next_index].is_populated()
        {
            completion_snapshots[next_index] = stats_block_from_candidate(cand);
        }

        if cand.llr < -LLR_TARGET {
            cand.is_active = false;
            if verbose {
                println!(
                    "!! FAILED {} (LLR {:.2} < -{:.2})",
                    cand.seed.name, cand.llr, LLR_TARGET
                );
            }
        } else if cand.llr > LLR_TARGET {
            if cand.elo >= best_elo {
                best_elo = cand.elo;
                best_candidate = Some(next_index);
            }
            if first_win_games.is_none() {
                first_win_games = Some(total_games);
            }
            if stop_at_first {
                break;
            }
            cand.is_active = false;
        }

        if verbose && (iteration % 50 == 0 || best_candidate.is_some()) {
            println!(
                "\n--- ITERATION {} | Total Games: {} ---",
                iteration, total_games
            );
            println!("-> Allocating games to: {}", cand.seed.name);
            let mut sorted = candidates.clone();
            sorted.sort_by(|a, b| {
                b.ucb_score()
                    .partial_cmp(&a.ucb_score())
                    .unwrap_or(Ordering::Equal)
            });
            for c in sorted {
                println!("{}", c);
            }
        }

        if complete_mode {
            let all_completed = completion_snapshots.iter().all(StatsBlock::is_populated);
            if all_completed {
                break;
            }
        } else if let Some(best_idx) = best_candidate {
            let best_elo = candidates[best_idx].elo;
            let best_threshold = if best_elo.is_nan() {
                f64::NEG_INFINITY
            } else {
                best_elo
            };
            let done = candidates
                .iter()
                .all(|c| !c.is_active || c.ucb_score() < best_threshold);
            if done {
                break;
            }
        }
    }

    let snapshots = std::array::from_fn(|idx| stats_block_from_candidate(&candidates[idx]));
    Ok((
        best_candidate,
        total_games,
        first_win_games,
        snapshots,
        completion_snapshots,
    ))
}

fn candidate_true_elos() -> [f64; NUM_CANDIDATES] {
    std::array::from_fn(|idx| pentanomial_true_elo(&CANDIDATE_DATA[idx].penta))
}

fn format_block(block: &StatsBlock) -> String {
    if !block.is_populated() {
        return "n/a".to_string();
    }
    let games = block.games.round() as usize;
    format!(
        "ELO={:>6.2} [{:>6.2},{:>6.2}] LLR={:>6.2} G={:>7}",
        block.elo, block.lcb, block.ucb, block.llr, games
    )
}

fn print_comparison_table(rows: &[ComparisonRow]) {
    let mut formatted: Vec<(String, String, String, String)> = Vec::with_capacity(rows.len());
    for row in rows {
        let name = row.name.to_string();
        let average = format_block(&row.average);
        let completion = format_block(&row.completion);
        let true_elo = if row.true_elo.is_finite() {
            format!("{:.2}", row.true_elo)
        } else {
            "n/a".to_string()
        };
        formatted.push((name, average, completion, true_elo));
    }

    let name_width = formatted
        .iter()
        .map(|row| row.0.len())
        .max()
        .unwrap_or(0)
        .max("Candidate".len());
    let average_width = formatted
        .iter()
        .map(|row| row.1.len())
        .max()
        .unwrap_or(0)
        .max("Average ELO/CI/LLR/Games".len());
    let completion_width = formatted
        .iter()
        .map(|row| row.2.len())
        .max()
        .unwrap_or(0)
        .max("Completion ELO/CI/LLR/Games".len());
    let true_width = formatted
        .iter()
        .map(|row| row.3.len())
        .max()
        .unwrap_or(0)
        .max("True Elo".len());

    println!(
        "{name:<name_width$} | {avg:<average_width$} | {comp:<completion_width$} | {true_elo:>true_width$}",
        name = "Candidate",
        avg = "Average ELO/CI/LLR/Games",
        comp = "Completion ELO/CI/LLR/Games",
        true_elo = "True Elo",
    );
    println!(
        "{:-<name_width$}-+-{:-<average_width$}-+-{:-<completion_width$}-+-{:-<true_width$}",
        "",
        "",
        "",
        "",
        name_width = name_width,
        average_width = average_width,
        completion_width = completion_width,
        true_width = true_width,
    );
    for (name, average, completion, true_elo) in formatted {
        println!(
            "{name:<name_width$} | {avg:<average_width$} | {comp:<completion_width$} | {true_elo:>true_width$}",
            name = name,
            avg = average,
            comp = completion,
            true_elo = true_elo,
        );
    }
}

pub fn run_monte_carlo(
    runs: usize,
    stopping: StoppingCondition,
    policy: AllocationPolicy,
    initial_tasks: usize,
) -> Result<()> {
    let initial_tasks = initial_tasks.max(1);
    let mut seed_rng = SmallRng::from_entropy();
    let seeds: Vec<u64> = (0..runs).map(|_| seed_rng.next_u64()).collect();

    let results: Vec<SimulationResult> = seeds
        .par_iter()
        .map(|seed| run_once_with_seed(false, stopping, policy, initial_tasks, *seed))
        .collect::<Result<Vec<_>>>()?;

    let mut total_sim = 0usize;
    let mut winner_counts = [0usize; NUM_CANDIDATES];
    let mut all_lower = 0usize;
    let mut average_totals = [zero_block(); NUM_CANDIDATES];
    let mut completion_totals = [zero_block(); NUM_CANDIDATES];
    let mut completion_counts = [0usize; NUM_CANDIDATES];
    let mut total_positive_completions = 0usize;
    let mut total_first_win_games = 0usize;
    let mut first_win_counts = 0usize;

    for result in &results {
        if let Some(idx) = result.winner_index {
            winner_counts[idx] += 1;
        } else {
            all_lower += 1;
        }
        total_sim += result.sim_games;
        if let Some(games) = result.games_to_first_win {
            total_first_win_games += games;
            first_win_counts += 1;
        }
        for i in 0..NUM_CANDIDATES {
            average_totals[i].accumulate(&result.average_stats[i]);
            let completion_block = result.completion_stats[i];
            if completion_block.is_populated() {
                completion_totals[i].accumulate(&completion_block);
                completion_counts[i] += 1;
                if completion_block.llr >= LLR_TARGET {
                    total_positive_completions += 1;
                }
            }
        }
    }

    println!("\n\n========================================");
    println!("      MONTE CARLO SUMMARY");
    println!("========================================");
    println!("Runs: {}", runs);
    println!("Initial tasks per candidate: {}", initial_tasks);
    let runs_f = runs as f64;
    let avg_sim = total_sim as f64 / runs_f;
    println!("Average simulation games: {:.0}", avg_sim);
    let avg_first_win_games = if first_win_counts > 0 {
        let value = total_first_win_games as f64 / first_win_counts as f64;
        println!("Average games to first winner: {:.0}", value);
        Some(value)
    } else {
        None
    };

    let true_elos = candidate_true_elos();
    let expected_true_elo: f64 = true_elos
        .iter()
        .enumerate()
        .map(|(idx, elo)| *elo * winner_counts[idx] as f64 / runs_f)
        .sum();
    println!(
        "Average true Elo of selected winner: {:.2}",
        expected_true_elo
    );
    if runs > 0 {
        let positive_avg = total_positive_completions as f64 / runs_f;
        let divisor = avg_first_win_games.unwrap_or(avg_sim);
        let positives_per_million_games = if divisor > 0.0 {
            (positive_avg / divisor) * 1_000_000.0
        } else {
            0.0
        };
        println!(
            "Average positive completions per run: {:.2} ({:.2} per M games)",
            positive_avg, positives_per_million_games
        );
    }

    if let Some(idx) = CANDIDATE_DATA
        .iter()
        .position(|seed| seed.name == "corradj4")
    {
        let count = winner_counts[idx];
        if count > 0 {
            println!(
                "corradj4 selected {}/{} times ({:.1}%)",
                count,
                runs,
                count as f64 / runs_f * 100.0
            );
        }
    }
    if all_lower > 0 {
        println!(
            "All candidates hit the lower bound in {}/{} runs ({:.1}%).",
            all_lower,
            runs,
            all_lower as f64 / runs_f * 100.0
        );
    }

    let mut distribution: Vec<(&'static str, usize)> = CANDIDATE_DATA
        .iter()
        .enumerate()
        .map(|(idx, seed)| (seed.name, winner_counts[idx]))
        .collect();
    if all_lower > 0 {
        distribution.push(("all_lower", all_lower));
    }
    distribution.retain(|(_, count)| *count > 0);
    if !distribution.is_empty() {
        println!("Winner distribution:");
        distribution.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, count) in distribution {
            println!(
                "  {:<10}: {} ({:.1}%)",
                name,
                count,
                count as f64 / runs_f * 100.0
            );
        }
    }
    println!();

    let avg_rows: Vec<ComparisonRow> = CANDIDATE_DATA
        .iter()
        .enumerate()
        .map(|(idx, seed)| ComparisonRow {
            name: seed.name,
            average: average_totals[idx].average(runs_f),
            completion: if completion_counts[idx] > 0 {
                completion_totals[idx].average(completion_counts[idx] as f64)
            } else {
                zero_block()
            },
            true_elo: true_elos[idx],
        })
        .collect();
    print_comparison_table(&avg_rows);
    Ok(())
}
