use statrs::distribution::{ContinuousCDF, Normal};

fn phi(x: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().cdf(x)
}

fn u_term(n: f64, gamma: f64, a: f64, y: f64) -> f64 {
    (2.0 * a * gamma * (std::f64::consts::PI * n * y / a).sin()
        - 2.0 * std::f64::consts::PI * n * (std::f64::consts::PI * n * y / a).cos())
        / (a.powi(2) * gamma.powi(2) + std::f64::consts::PI.powi(2) * n.powi(2))
}

#[derive(Clone, Copy)]
pub struct Brownian {
    pub a: f64,
    pub b: f64,
    pub mu: f64,
    sigma2: f64,
}

impl Brownian {
    pub fn new(a: f64, b: f64, mu: f64, sigma: f64) -> Self {
        Self {
            a,
            b,
            mu,
            sigma2: sigma * sigma,
        }
    }

    pub fn outcome_cdf(&self, t: f64, y: f64) -> f64 {
        let sigma2 = self.sigma2;
        let gamma = self.mu / sigma2;
        let a = self.a;
        let b = self.b;
        let span = b - a;
        let condition = sigma2 * t / span.powi(2) < 1e-2 || (gamma * span).abs() > 15.0;
        let ret = if condition {
            self.outcome_cdf_alt2(t, y)
        } else {
            self.outcome_cdf_alt1(t, y)
        };
        ret
    }

    fn outcome_cdf_alt1(&self, t: f64, y: f64) -> f64 {
        let mu = self.mu;
        let sigma2 = self.sigma2;
        let a = self.a;
        let b = self.b;
        let span = b - a;
        let x = -a;
        let y_shifted = y - a;
        let gamma = mu / sigma2;
        let mut n = 1.0;
        let mut s = 0.0;
        let lambda1 =
            ((std::f64::consts::PI / span).powi(2)) * sigma2 / 2.0 + (mu * mu / sigma2) / 2.0;
        let t0 = (-lambda1 * t - x * gamma + y_shifted * gamma).exp();
        loop {
            let lambda_n = ((n * std::f64::consts::PI / span).powi(2)) * sigma2 / 2.0
                + (mu * mu / sigma2) / 2.0;
            let t1 = (-(lambda_n - lambda1) * t).exp();
            let t3 = u_term(n, gamma, span, y_shifted);
            let t4 = (n * std::f64::consts::PI * x / span).sin();
            s += t1 * t3 * t4;
            if (t0 * t1 * t3).abs() <= 1e-9 || n > 10_000.0 {
                break;
            }
            n += 1.0;
        }
        let pre = if gamma * span > 30.0 {
            (-2.0 * gamma * x).exp()
        } else if (gamma * span).abs() < 1e-8 {
            (span - x) / span
        } else {
            (1.0 - (2.0 * gamma * (span - x)).exp()) / (1.0 - (2.0 * gamma * span).exp())
        };
        pre + t0 * s
    }

    fn outcome_cdf_alt2(&self, t: f64, y: f64) -> f64 {
        let denom = (t * self.sigma2).sqrt();
        let offset = self.mu * t;
        let gamma = self.mu / self.sigma2;
        let a = self.a;
        let b = self.b;
        let z = (y - offset) / denom;
        let za = (-y + offset + 2.0 * a) / denom;
        let zb = (y - offset - 2.0 * b) / denom;
        let t1 = phi(z);
        let t2 = if gamma * a >= 5.0 {
            -((-(za * za) / 2.0).exp() * (2.0 * gamma * a).exp())
                / (2.0 * std::f64::consts::PI).sqrt()
                * (1.0 / za - 1.0 / za.powi(3))
        } else {
            (2.0 * gamma * a).exp() * phi(za)
        };
        let t3 = if gamma * b >= 5.0 {
            -((-(zb * zb) / 2.0).exp() * (2.0 * gamma * b).exp())
                / (2.0 * std::f64::consts::PI).sqrt()
                * (1.0 / zb - 1.0 / zb.powi(3))
        } else {
            (2.0 * gamma * b).exp() * phi(zb)
        };
        t1 + t2 - t3
    }
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
    fn outcome_cdf_matches_python_alt1_branch() {
        // Reference computed from multitakes_simul_py/core/brownian.py::Brownian.
        let brownian = Brownian::new(-1.0, 1.0, 0.0, 0.5);
        let value = brownian.outcome_cdf(1.0, 0.3);
        assert_close(value, 0.725_412_065_439_910_9_f64, 1e-12);
    }

    #[test]
    fn outcome_cdf_matches_python_alt2_branch() {
        // Forces the Siegmund approximation branch via high |gamma * span|.
        let brownian = Brownian::new(-0.5, 1.5, 0.01, 0.02);
        let value = brownian.outcome_cdf(5.0, 0.2);
        assert_close(value, 0.999_601_884_921_204_6_f64, 1e-12);
    }
}
