use anyhow::{Result, bail};

use crate::brownian::Brownian;
use crate::llr::{
    self, NELO_DIVIDED_BY_NT, llr_drift_variance_alt2, llr_logistic, llr_normalized, logistic,
};
use crate::root::brentq;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EloModel {
    Logistic,
    Normalized,
    BayesElo,
}

#[derive(Clone, Copy, Debug)]
pub struct SprtParams {
    pub alpha: f64,
    pub beta: f64,
    pub elo0: f64,
    pub elo1: f64,
    pub elo_model: EloModel,
    pub p: f64,
}

#[derive(Clone, Debug)]
pub struct SprtAnalysis {
    pub elo: f64,
    pub ci: (f64, f64),
}

pub struct Sprt {
    params: SprtParams,
    lower_bound: f64,
    upper_bound: f64,
    pdf: Vec<(f64, f64)>,
    sigma_pg: f64,
    s0: f64,
    s1: f64,
    llr: f64,
    t: f64,
    clamped: bool,
}

impl Sprt {
    pub fn new(params: SprtParams) -> Self {
        let lower_bound = (params.beta / (1.0 - params.alpha)).ln();
        let upper_bound = ((1.0 - params.beta) / params.alpha).ln();
        Self {
            params,
            lower_bound,
            upper_bound,
            pdf: Vec::new(),
            sigma_pg: 0.0,
            s0: 0.0,
            s1: 0.0,
            llr: 0.0,
            t: 0.0,
            clamped: false,
        }
    }

    fn elo_to_score(&self, elo: f64) -> f64 {
        match self.params.elo_model {
            EloModel::Logistic | EloModel::BayesElo => logistic(elo),
            EloModel::Normalized => {
                let nt = elo / NELO_DIVIDED_BY_NT;
                nt * self.sigma_pg + 0.5
            }
        }
    }

    pub fn set_state(&mut self, results: &[u64]) -> Result<()> {
        let (n, pdf) = llr::results_to_pdf(results)?;
        self.pdf = pdf;
        if matches!(self.params.elo_model, EloModel::Normalized) {
            let (_mu, var) = llr::stats(&self.pdf);
            self.sigma_pg = match results.len() {
                5 => (2.0 * var).sqrt(),
                3 => var.sqrt(),
                len => bail!("Unsupported results length {}", len),
            };
        }
        self.s0 = self.elo_to_score(self.params.elo0);
        self.s1 = self.elo_to_score(self.params.elo1);
        let (mu_llr, _var_llr) = llr_drift_variance_alt2(&self.pdf, self.s0, self.s1, None);
        self.llr = n * mu_llr;
        self.t = n;
        let slope = if n > 0.0 { self.llr / n } else { 0.0 };
        self.clamped = self.llr > 1.03 * self.upper_bound || self.llr < 1.03 * self.lower_bound;
        if slope != 0.0 {
            if self.llr < self.lower_bound {
                self.t = self.lower_bound / slope;
                self.llr = self.lower_bound;
            } else if self.llr > self.upper_bound {
                self.t = self.upper_bound / slope;
                self.llr = self.upper_bound;
            }
        }
        Ok(())
    }

    fn outcome_prob(&self, elo: f64) -> f64 {
        let s = logistic(elo);
        let (mu_llr, var_llr) = llr_drift_variance_alt2(&self.pdf, self.s0, self.s1, Some(s));
        let sigma_llr = var_llr.sqrt().max(1e-12);
        Brownian::new(self.lower_bound, self.upper_bound, mu_llr, sigma_llr)
            .outcome_cdf(self.t, self.llr)
    }

    fn lower_cb(&self, p: f64) -> Result<f64> {
        let avg_elo = (self.params.elo0 + self.params.elo1) / 2.0;
        let delta = self.params.elo1 - self.params.elo0;
        let mut n = 30.0;
        let target = 1.0 - p;
        loop {
            let elo_min = (avg_elo - n * delta).max(-1000.0);
            let elo_max = (avg_elo + n * delta).min(1000.0);
            let f = |elo: f64| self.outcome_prob(elo) - target;
            match brentq(elo_min, elo_max, f, 1e-9, 200) {
                Ok(sol) => return Ok(sol),
                Err(_) => {
                    if elo_min > -1000.0 || elo_max < 1000.0 {
                        n *= 2.0;
                        continue;
                    }
                    let prob_min = self.outcome_prob(elo_min) - target;
                    if prob_min > 0.0 {
                        return Ok(elo_max);
                    } else {
                        return Ok(elo_min);
                    }
                }
            }
        }
    }

    pub fn analytics(&self, p: f64) -> Result<SprtAnalysis> {
        let elo = self.lower_cb(0.5)?;
        let ci_lower = self.lower_cb(p / 2.0)?;
        let ci_upper = self.lower_cb(1.0 - p / 2.0)?;
        Ok(SprtAnalysis {
            elo,
            ci: (ci_lower, ci_upper),
        })
    }

    pub fn quantile(&self, p: f64) -> Result<f64> {
        self.lower_cb(p)
    }
}

#[derive(Clone, Debug)]
pub struct SprtResult {
    pub elo: f64,
    pub ci: (f64, f64),
    pub llr: f64,
}

pub fn sprt_elo(pentanomial: &[u64], params: &SprtParams) -> Result<SprtResult> {
    let mut sprt = Sprt::new(*params);
    sprt.set_state(pentanomial)?;
    let analysis = sprt.analytics(params.p)?;
    let llr_value = match params.elo_model {
        EloModel::Logistic | EloModel::BayesElo => {
            llr_logistic(params.elo0, params.elo1, pentanomial)?
        }
        EloModel::Normalized => llr_normalized(params.elo0, params.elo1, pentanomial)?,
    };
    Ok(SprtResult {
        elo: analysis.elo,
        ci: analysis.ci,
        llr: llr_value,
    })
}

pub fn sprt_quantile(pentanomial: &[u64], params: &SprtParams, percentile: f64) -> Result<f64> {
    let mut sprt = Sprt::new(*params);
    sprt.set_state(pentanomial)?;
    sprt.quantile(percentile)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PARAMS: SprtParams = SprtParams {
        alpha: 0.05,
        beta: 0.05,
        elo0: 0.0,
        elo1: 2.0,
        elo_model: EloModel::Normalized,
        p: 0.05,
    };

    #[test]
    fn sprt_matches_python_example_one() {
        let data = [461u64, 16_726, 36_723, 16_920, 514];
        let stats = sprt_elo(&data, &TEST_PARAMS).unwrap();
        assert!((stats.elo - 0.7175629178).abs() < 1e-6);
        assert!((stats.ci.0 - (-0.2243908804)).abs() < 1e-6);
        assert!((stats.ci.1 - 1.6490993951).abs() < 1e-6);
        assert!((stats.llr - 1.0025304976).abs() < 1e-6);
    }

    #[test]
    fn sprt_matches_python_example_two() {
        let data = [82u64, 2_646, 5_623, 2_507, 70];
        let stats = sprt_elo(&data, &TEST_PARAMS).unwrap();
        assert!((stats.elo - (-2.5908198747)).abs() < 1e-6);
        assert!((stats.ci.0 - (-4.9570312405)).abs() < 1e-6);
        assert!((stats.ci.1 - (-0.2269759106)).abs() < 1e-6);
        assert!((stats.llr - (-2.1897352413)).abs() < 1e-6);
    }
}
