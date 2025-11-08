pub fn pentanomial_expected_score(penta: &[u64; 5]) -> f64 {
    let weights = [0.0, 0.5, 1.0, 1.5, 2.0];
    let total_pairs: f64 = penta.iter().map(|&v| v as f64).sum();
    if total_pairs == 0.0 {
        return 0.5;
    }
    let total_score: f64 = weights
        .iter()
        .zip(penta.iter())
        .map(|(w, c)| w * (*c as f64))
        .sum();
    total_score / (2.0 * total_pairs)
}

pub fn score_to_logistic_elo(score: f64) -> f64 {
    let epsilon = 1e-6;
    let s = score.clamp(epsilon, 1.0 - epsilon);
    400.0 * (s / (1.0 - s)).log10()
}

pub fn pentanomial_true_elo(penta: &[u64; 5]) -> f64 {
    score_to_logistic_elo(pentanomial_expected_score(penta))
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
    fn expected_score_handles_zero_pairs() {
        let penta = [0, 0, 0, 0, 0];
        assert_close(pentanomial_expected_score(&penta), 0.5, 1e-12);
    }

    #[test]
    fn expected_score_matches_python_reference_cases() {
        let cases = [
            (
                [461, 16_726, 36_723, 16_920, 514],
                0.501_051_244_673_693_7_f64,
            ),
            ([82, 2_646, 5_623, 2_507, 70], 0.496_271_046_852_123_f64),
            (
                [346, 12_061, 26_295, 12_166, 364],
                0.500_688_046_533_416_6_f64,
            ),
            (
                [159, 5_353, 11_766, 5_551, 163],
                0.502_239_909_533_750_9_f64,
            ),
        ];
        for (penta, expected) in cases {
            assert_close(pentanomial_expected_score(&penta), expected, 1e-12);
        }
    }

    #[test]
    fn true_elo_matches_python_reference_cases() {
        let cases = [
            (
                [461, 16_726, 36_723, 16_920, 514],
                0.730_480_693_819_749_8_f64,
            ),
            ([82, 2_646, 5_623, 2_507, 70], -2.591_190_082_331_106_5_f64),
            (
                [346, 12_061, 26_295, 12_166, 364],
                0.478_104_002_193_311_1_f64,
            ),
            ([159, 5_353, 11_766, 5_551, 163], 1.556_458_972_885_468_f64),
        ];
        for (penta, expected) in cases {
            assert_close(pentanomial_true_elo(&penta), expected, 1e-9);
        }
    }
}
