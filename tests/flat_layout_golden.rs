//! Golden snapshot guarding the nested -> flat tensor layout refactor in `compression::innovative`.
//! Outputs are compared bit-for-bit against values recorded from the previous implementation.

use kortex::compression::{
    CompressionQualityReport, OpdmRopeEvaluator, OrthogonalHolographyEvaluator, SvdHologramEvaluator,
};

fn fingerprint(r: &CompressionQualityReport) -> String {
    format!(
        "{:016x}|{:016x}|{:016x}|{:08x}|{:016x}|{:016x}",
        r.mean_cosine_similarity.to_bits(),
        r.relative_frobenius_error.to_bits(),
        r.compression_ratio.to_bits(),
        r.max_absolute_error.to_bits(),
        r.token_agreement_pct.to_bits(),
        r.innovation_score.to_bits(),
    )
}

fn all_fingerprints() -> Vec<(&'static str, String)> {
    vec![
        ("svd_r16_n512_d64", fingerprint(&SvdHologramEvaluator::new(16, 512, 64, 6144).evaluate())),
        ("svd_r8_n300_d128", fingerprint(&SvdHologramEvaluator::new(8, 300, 128, 6144).evaluate())),
        ("ortho_l8_n256_d64", fingerprint(&OrthogonalHolographyEvaluator::new(8, 256, 64, 6144).evaluate())),
        ("ortho_l4_n100_d128", fingerprint(&OrthogonalHolographyEvaluator::new(4, 100, 128, 6144).evaluate())),
        ("opdm_p4_d64_n512", fingerprint(&OpdmRopeEvaluator::new(4, 64, 512, 6144).evaluate())),
        ("opdm_p8_d128_n256", fingerprint(&OpdmRopeEvaluator::new(8, 128, 256, 6144).evaluate())),
    ]
}

const EXPECTED: &[(&str, &str)] = &[
    ("svd_r16_n512_d64", "3feffec830f89474|3f91a84eaa320085|400c71c71c71c71c|3d4ccc80|4058f33333333333|407a94071938170f"),
    ("svd_r8_n300_d128", "3feffd90631d1247|3f98f86b253de6bd|40266e0e5aea77a0|3d4ccc80|4058f33333333333|408f019d3f8efb79"),
    ("ortho_l8_n256_d64", "3fd6a09413785480|40052a7762a371f5|4020000000000000|40b9beb4|4011ad638d86430b|400102dff4ecee5d"),
    ("ortho_l4_n100_d128", "3fdfff7e73bb3546|3ffbb6301eadb0df|4010000000000000|403a4802|4028fed0640bf158|401c816e31741e30"),
    ("opdm_p4_d64_n512", "3fe012584e8fe062|3ffbcbadc567f060|4010000000000000|403f1458|40295660ae3a19d4|4018efceabd2053d"),
    ("opdm_p8_d128_n256", "3fd6ca6725aae891|40054c04897f6a62|4020000000000000|40d99a47|4012101f870d7ec8|3ffdff0a75bd3115"),
];

#[test]
fn test_flat_layout_matches_golden() {
    let fps = all_fingerprints();
    assert_eq!(fps.len(), EXPECTED.len());
    for (actual, expected) in fps.iter().zip(EXPECTED.iter()) {
        assert_eq!(actual.0, expected.0);
        assert_eq!(
            actual.1, expected.1,
            "Fingerprint mismatch for {}: expected {}, got {}",
            expected.0, expected.1, actual.1
        );
    }
}
