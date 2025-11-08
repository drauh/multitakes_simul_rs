use anyhow::{Result, anyhow, bail};

/// Simple bisection-based root finder used as a stand-in for scipy.optimize.brentq.
pub fn brentq<F>(a: f64, b: f64, f: F, tol: f64, max_iter: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let fa = f(a);
    let fb = f(b);

    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa.signum() == fb.signum() {
        bail!("Root is not bracketed: f(a) and f(b) share the same sign");
    }

    let mut left = a;
    let mut right = b;
    let mut f_left = fa;

    for _ in 0..max_iter {
        let mid = 0.5 * (left + right);
        let f_mid = f(mid);
        if f_mid.abs() <= tol || 0.5 * (right - left).abs() < tol {
            return Ok(mid);
        }
        if f_left.signum() == f_mid.signum() {
            left = mid;
            f_left = f_mid;
        } else {
            right = mid;
        }
    }

    Err(anyhow!(
        "Root finder failed to converge within {} iterations",
        max_iter
    ))
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
    fn matches_scipy_brentq_cos_minus_x_example() {
        // Reference root computed via SciPy's brentq on cos(x) - x.
        let root = brentq(0.0, 1.0, |x| x.cos() - x, 1e-12, 200).unwrap();
        assert_close(root, 0.739_085_133_215_160_7_f64, 1e-12);
    }

    #[test]
    fn finds_cubic_root_with_endpoint_detection() {
        // f(∛2) = 0; the function changes sign across [0, 2].
        let root = brentq(0.0, 2.0, |x| x.powi(3) - 2.0, 1e-12, 200).unwrap();
        assert_close(root, 1.259_921_049_894_873_2_f64, 1e-12);
    }

    #[test]
    fn errors_when_interval_not_bracketing_root() {
        let res = brentq(0.0, 1.0, |x| x * x + 1.0, 1e-9, 50);
        assert!(res.is_err());
    }
}
