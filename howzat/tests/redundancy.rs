#[path = "support/common.rs"]
mod common;

use common::parse_cdd_file;

#[test]
fn implicit_and_strong_redundancy_for_samplev3() {
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplev3.ext",
    ))
    .expect("parse samplev3");
    let impl_lin = parsed
        .matrix
        .implicit_linearity_rows()
        .expect("implicit linearity");
    assert!(
        !impl_lin.is_empty(),
        "expected some implicit linearity rows in samplev3"
    );
    let strong = parsed
        .matrix
        .strongly_redundant_rows()
        .expect("strong redundancy");
    assert!(
        strong.cardinality() > 0,
        "expected some strongly redundant rows in samplev3"
    );
}

#[test]
fn redundancy_checks_on_small_h_examples() {
    for name in [
        "sampleh5.ine",
        "sampleh6.ine",
        "sampleh7.ine",
        "sampleh8.ine",
    ] {
        let parsed = parse_cdd_file(&std::path::PathBuf::from(format!(
            "tests/data/examples/{}",
            name
        )))
        .expect("parse sampleh");
        parsed.matrix.redundant_rows().expect("redundant rows");
        parsed
            .matrix
            .implicit_linearity_rows()
            .expect("implicit linearity");
    }
}
