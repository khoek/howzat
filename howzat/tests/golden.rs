#[path = "support/common.rs"]
mod common;

use hullabaloo::types::RowSet;

use common::{format_matrix, matrices_approx_equal, normalize_matrix, parse_cdd_file};

#[test]
fn bug45_redundancy_matches_expected_canonical_form() {
    let input = parse_cdd_file(&std::path::PathBuf::from("tests/data/examples/bug45.ine"))
        .expect("parse bug45 input");
    let expected = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/bug45res.ine",
    ))
    .expect("parse bug45 result");

    let (reduced, _impl_lin, redset, _) =
        match input.matrix.canonicalize().expect("canonicalize bug45") {
            common::ParsedCanonicalizationResult::Inequality(r) => (
                r.matrix().clone(),
                r.implicit_linearity().clone(),
                r.redundant_rows().clone(),
                r.positions().clone(),
            ),
            _ => panic!("expected inequality result"),
        };
    let norm_reduced = normalize_matrix(&reduced);
    let norm_expected =
        normalize_matrix(expected.matrix.as_inequality().expect("inequality matrix"));
    let mut expected_red = RowSet::new(5);
    expected_red.insert(2);
    expected_red.insert(3);
    expected_red.insert(4);

    let left_indices: Vec<usize> = (0..redset.len()).filter(|&i| redset.contains(i)).collect();
    let right_indices: Vec<usize> = (0..expected_red.len())
        .filter(|&i| expected_red.contains(i))
        .collect();
    assert_eq!(
        left_indices, right_indices,
        "canonicalization should remove redundant rows 2,3,4"
    );

    assert!(
        matrices_approx_equal(&norm_reduced, &norm_expected, 1e-9),
        "canonicalized bug45 does not match expected result\ncomputed:\n{}\nexpected:\n{}",
        format_matrix(&norm_reduced),
        format_matrix(&norm_expected)
    );
}
