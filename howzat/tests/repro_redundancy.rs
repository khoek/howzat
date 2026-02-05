use calculo::num::Num;
use howzat::matrix::LpMatrix as Matrix;
use hullabaloo::types::Inequality;

#[path = "support/common.rs"]
mod common;

#[test]
fn test_trivial_redundancy() {
    common::init_logging();
    let eps = f64::default_eps();
    // Matrix:
    // 0: x_3 >= 0
    // 1: x_3 >= 1
    // 2: 1 >= 0 (trivial)
    let rows = vec![
        vec![0.0, 0.0, 0.0, 1.0],
        vec![-1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 0.0],
    ];
    let matrix = Matrix::<f64, Inequality>::from_rows(rows);

    // Row 2 (1 >= 0) should be redundant.
    // Row 0 (x_3 >= 0) is implied by Row 1 (x_3 >= 1). So Row 0 is redundant.
    // Row 1 is essential.

    let redundant = matrix.redundant_rows(&eps).unwrap();

    assert!(redundant.contains(2), "Row 2 (1 >= 0) should be redundant");
    assert!(
        redundant.contains(0),
        "Row 0 (x_3 >= 0) should be redundant"
    );
    assert!(
        !redundant.contains(1),
        "Row 1 (x_3 >= 1) should NOT be redundant"
    );
}
