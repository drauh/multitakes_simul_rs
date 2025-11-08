use crate::root::brentq;
use anyhow::{Result, bail};

pub const NELO_DIVIDED_BY_NT: f64 = 800.0 / std::f64::consts::LN_10; // 347.4355855...
const EPSILON: f64 = 1e-3;

#[derive(Clone, Copy)]
pub enum Statistic {
    Expectation,
    TValue,
}

pub fn logistic(elo: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf(-elo / 400.0))
}

pub fn regularize(counts: &[f64]) -> Vec<f64> {
    counts
        .iter()
        .map(|&c| if c == 0.0 { EPSILON } else { c })
        .collect()
}

pub fn results_to_pdf(results: &[u64]) -> Result<(f64, Vec<(f64, f64)>)> {
    if results.is_empty() {
        bail!("results cannot be empty");
    }
    let sum: f64 = results.iter().map(|&v| v as f64).sum();
    if sum == 0.0 {
        bail!("results sum to zero");
    }
    let count = results.len();
    let mut freq: Vec<f64> = results.iter().map(|&v| v as f64).collect();
    freq = regularize(&freq);
    let n: f64 = freq.iter().sum();
    let pdf: Vec<(f64, f64)> = freq
        .iter()
        .enumerate()
        .map(|(i, &v)| ((i as f64) / ((count - 1) as f64), v / n))
        .collect();
    Ok((n, pdf))
}

pub fn stats(pdf: &[(f64, f64)]) -> (f64, f64) {
    let s: f64 = pdf.iter().map(|(value, prob)| value * prob).sum();
    let var: f64 = pdf
        .iter()
        .map(|(value, prob)| prob * (value - s).powi(2))
        .sum();
    (s, var)
}

fn secular(pdf: &[(f64, f64)]) -> Result<f64> {
    let epsilon = 1e-9;
    let min_a = pdf.iter().map(|(a, _)| *a).fold(f64::INFINITY, f64::min);
    let max_a = pdf
        .iter()
        .map(|(a, _)| *a)
        .fold(f64::NEG_INFINITY, f64::max);
    if min_a * max_a >= 0.0 {
        bail!("secular equation requires values of mixed sign");
    }
    let lower_bound = -1.0 / max_a;
    let upper_bound = -1.0 / min_a;

    let f = |x: f64| -> f64 {
        pdf.iter()
            .map(|(ai, pi)| pi * ai / (1.0 + x * ai))
            .sum::<f64>()
    };

    brentq(lower_bound + epsilon, upper_bound - epsilon, f, 1e-12, 200)
}

fn mle_expected(pdfhat: &[(f64, f64)], s: f64) -> Result<Vec<(f64, f64)>> {
    let pdf1: Vec<(f64, f64)> = pdfhat.iter().map(|(ai, pi)| (ai - s, *pi)).collect();
    let x = secular(&pdf1)?;
    let pdf_mle: Vec<(f64, f64)> = pdfhat
        .iter()
        .map(|(ai, pi)| (*ai, pi / (1.0 + x * (*ai - s))))
        .collect();
    let (s_check, _) = stats(&pdf_mle);
    assert!((s - s_check).abs() < 1e-6);
    Ok(pdf_mle)
}

fn uniform(pdfhat: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = pdfhat.len() as f64;
    pdfhat.iter().map(|(ai, _)| (*ai, 1.0 / n)).collect()
}

fn mle_t_value(pdfhat: &[(f64, f64)], reference: f64, s: f64) -> Result<Vec<(f64, f64)>> {
    let n = pdfhat.len();
    let mut pdf_mle = uniform(pdfhat);
    for _ in 0..10 {
        let previous = pdf_mle.clone();
        let (mu, var) = stats(&pdf_mle);
        let sigma = var.sqrt().max(1e-12);
        let pdf1: Vec<(f64, f64)> = pdfhat
            .iter()
            .map(|(ai, pi)| {
                let adjustment =
                    ai - reference - s * sigma * (1.0 + ((mu - ai) / sigma).powi(2)) / 2.0;
                (adjustment, *pi)
            })
            .collect();
        let x = secular(&pdf1)?;
        pdf_mle = (0..n)
            .map(|i| {
                let (ai, pi_hat) = pdfhat[i];
                let denom = 1.0 + x * pdf1[i].0;
                (ai, pi_hat / denom)
            })
            .collect();
        let delta = pdf_mle
            .iter()
            .zip(previous.iter())
            .map(|((_, new_p), (_, old_p))| (new_p - old_p).abs())
            .fold(0.0, f64::max);
        if delta < 1e-9 {
            break;
        }
    }
    let (mu, var) = stats(&pdf_mle);
    assert!((s - (mu - reference) / var.sqrt()).abs() < 1e-5);
    Ok(pdf_mle)
}

fn llr_jumps(
    pdf: &[(f64, f64)],
    s0: f64,
    s1: f64,
    reference: Option<f64>,
    statistic: Statistic,
) -> Result<Vec<(f64, f64)>> {
    let (pdf0, pdf1) = match statistic {
        Statistic::Expectation => (mle_expected(pdf, s0)?, mle_expected(pdf, s1)?),
        Statistic::TValue => {
            let ref_value = reference.unwrap_or(0.5);
            (
                mle_t_value(pdf, ref_value, s0)?,
                mle_t_value(pdf, ref_value, s1)?,
            )
        }
    };
    let jumps = pdf
        .iter()
        .enumerate()
        .map(|(i, (_, prob))| {
            let value = pdf1[i].1.ln() - pdf0[i].1.ln();
            (value, *prob)
        })
        .collect();
    Ok(jumps)
}

fn llr(
    pdf: &[(f64, f64)],
    s0: f64,
    s1: f64,
    reference: Option<f64>,
    statistic: Statistic,
) -> Result<f64> {
    let jumps = llr_jumps(pdf, s0, s1, reference, statistic)?;
    Ok(stats(&jumps).0)
}

pub fn llr_logistic(elo0: f64, elo1: f64, results: &[u64]) -> Result<f64> {
    let s0 = logistic(elo0);
    let s1 = logistic(elo1);
    let (n, pdf) = results_to_pdf(results)?;
    Ok(n * llr(&pdf, s0, s1, None, Statistic::Expectation)?)
}

pub fn llr_normalized(nelo0: f64, nelo1: f64, results: &[u64]) -> Result<f64> {
    let nt0 = nelo0 / NELO_DIVIDED_BY_NT;
    let nt1 = nelo1 / NELO_DIVIDED_BY_NT;
    let sqrt2 = 2.0_f64.sqrt();
    let (t0, t1) = match results.len() {
        3 => (nt0, nt1),
        5 => (nt0 * sqrt2, nt1 * sqrt2),
        len => bail!("Unsupported results length {}", len),
    };
    let (n, pdf) = results_to_pdf(results)?;
    Ok(n * llr(&pdf, t0, t1, Some(0.5), Statistic::TValue)?)
}

pub fn llr_drift_variance_alt2(pdf: &[(f64, f64)], s0: f64, s1: f64, s: Option<f64>) -> (f64, f64) {
    let (s_emp, var_emp) = stats(pdf);
    let (s_true, v) = if let Some(target) = s {
        (target, var_emp + (target - s_emp).powi(2))
    } else {
        (s_emp, var_emp)
    };
    let mu = (s_true - (s0 + s1) / 2.0) * (s1 - s0) / v;
    let var = (s1 - s0).powi(2) / v;
    (mu, var)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(value: f64, expected: f64, tol: f64) {
        assert!(
            (value - expected).abs() <= tol,
            "value {value} differed from {expected} (tol {tol})"
        );
    }

    #[test]
    fn llr_logistic_matches_fishtest_pentanomials() {
        // Reference numbers taken from multitakes_simul_py/core/LLRcalc.py (fishtest upstream).
        let cases = [
            (
                [461, 16_726, 36_723, 16_920, 514],
                -2.421_220_130_461_109_f64,
            ),
            ([82, 2_646, 5_623, 2_507, 70], -4.932_611_354_708_9845_f64),
        ];

        for (results, expected) in cases {
            let llr = llr_logistic(0.0, 2.0, &results).unwrap();
            assert_close(llr, expected, 1e-9);
        }
    }

    #[test]
    fn llr_normalized_matches_fishtest_pentanomials() {
        // Same datasets as above; normalized-elo values generated via LLRcalc.LLR_normalized.
        let cases = [
            (
                [461, 16_726, 36_723, 16_920, 514],
                1.002_530_497_672_872_8_f64,
            ),
            ([82, 2_646, 5_623, 2_507, 70], -2.189_735_241_282_270_4_f64),
        ];

        for (results, expected) in cases {
            let llr = llr_normalized(0.0, 2.0, &results).unwrap();
            assert_close(llr, expected, 1e-9);
        }
    }

    #[test]
    fn llr_matches_python_three_outcome_reference() {
        // Deterministic ternary test copied from fishtest-style python helpers.
        let results = [1_300, 400, 1_500];
        let logistic_llr = llr_logistic(-1.0, 3.0, &results).unwrap();
        let normalized_llr = llr_normalized(-1.0, 3.0, &results).unwrap();

        assert_close(logistic_llr, 2.510_409_338_650_599_f64, 1e-9);
        assert_close(normalized_llr, 2.355_527_942_566_393_3_f64, 1e-9);
    }

    #[test]
    fn drift_variance_alt2_matches_python_reference() {
        let results = [461, 16_726, 36_723, 16_920, 514];
        let (_n, pdf) = results_to_pdf(&results).unwrap();
        let s0 = logistic(0.0);
        let s1 = logistic(2.0);
        let (mu, var) = llr_drift_variance_alt2(&pdf, s0, s1, None);

        assert_close(mu, -3.394_054_254_397_120_6e-5, 1e-12);
        assert_close(var, 2.518_663_585_070_327e-4, 1e-12);
    }
}
