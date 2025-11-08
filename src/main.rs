mod brownian;
mod llr;
mod pentanomial;
mod root;
mod simulation;
mod sprt;

use clap::{Parser, ValueEnum};
use simulation::{AllocationPolicy, DEFAULT_INITIAL_TASKS, StoppingCondition, run_monte_carlo};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum PolicyArg {
    #[value(alias = "ucb", alias = "upper-confidence-bound")]
    Ucb,
    #[value(alias = "ts", alias = "thompson", alias = "thompson-sampling")]
    Thompson,
    #[value(alias = "seq", alias = "sequential", alias = "round-robin")]
    Sequential,
}

impl From<PolicyArg> for AllocationPolicy {
    fn from(arg: PolicyArg) -> Self {
        match arg {
            PolicyArg::Ucb => AllocationPolicy::Ucb,
            PolicyArg::Thompson => AllocationPolicy::ThompsonSampling,
            PolicyArg::Sequential => AllocationPolicy::Sequential,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum StoppingArg {
    #[value(
        alias = "default",
        alias = "standard",
        alias = "prune",
        alias = "boundary-prune",
        alias = "prune-on-boundary"
    )]
    UcbDominance,
    #[value(alias = "stop-at-first", alias = "first-hit")]
    StopAtFirst,
    #[value(alias = "full", alias = "sequential", alias = "run-to-boundaries")]
    Complete,
}

impl From<StoppingArg> for StoppingCondition {
    fn from(arg: StoppingArg) -> Self {
        match arg {
            StoppingArg::UcbDominance => StoppingCondition::UcbDominance,
            StoppingArg::StopAtFirst => StoppingCondition::StopAtFirst,
            StoppingArg::Complete => StoppingCondition::Complete,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Monte Carlo simulator for multi-take SPRT scheduling",
    long_about = None
)]
struct Args {
    /// Number of Monte Carlo repetitions (default 1)
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// Stopping rule applied when a candidate hits a SPRT boundary (or run everyone to completion with `complete`)
    #[arg(long, value_enum, default_value_t = StoppingArg::UcbDominance)]
    stopping: StoppingArg,

    /// Allocation policy to use for selecting the next candidate (ucb|ts)
    #[arg(long, value_enum, default_value_t = PolicyArg::Ucb)]
    policy: PolicyArg,

    /// Number of fixed-size tasks to assign to every candidate before policies kick in
    #[arg(
        long,
        value_name = "TASKS",
        default_value_t = DEFAULT_INITIAL_TASKS,
        value_parser = clap::value_parser!(usize)
    )]
    initial_tasks: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let runs = args.runs.max(1);
    run_monte_carlo(
        runs,
        StoppingCondition::from(args.stopping),
        AllocationPolicy::from(args.policy),
        args.initial_tasks,
    )?;

    Ok(())
}
