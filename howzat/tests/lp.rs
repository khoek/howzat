#[path = "support/common.rs"]
mod common;

use calculo::num::Num;
use howzat::lp::LpProblem;
use howzat::lp::{LpObjective, LpSolver, LpStatus};
use howzat::matrix::LpMatrixBuilder;

use common::{TestNumber, parse_cdd_file};

const SOLVERS: [LpSolver; 2] = [LpSolver::DualSimplex, LpSolver::CrissCross];

fn approx_eq(a: TestNumber, b: TestNumber, tol: TestNumber) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn solves_testlp2_example() {
    let eps = f64::default_eps();
    let matrix = LpMatrixBuilder::<f64, hullabaloo::types::Inequality>::from_rows(vec![
        vec![4.0 / 3.0, -2.0, -1.0],
        vec![2.0 / 3.0, 0.0, -1.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ])
    .with_row_vec(vec![0.0, 3.0, 4.0])
    .with_objective(LpObjective::Maximize)
    .build();

    for solver in SOLVERS {
        let lp = LpProblem::from_matrix(&matrix, &eps).expect("build lp");
        let solution = lp.solve(solver, &eps);
        assert_eq!(
            solution.status(),
            LpStatus::Optimal,
            "{solver:?} failed on testlp2 example"
        );
        assert!(
            solution.optimal_value().is_sign_positive()
                || approx_eq(*solution.optimal_value(), 0.0, 1e-9),
            "{solver:?} produced non-positive optimum for testlp2 example"
        );
    }
}

#[test]
fn solves_samplelp_input() {
    let eps = f64::default_eps();
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplelp.ine",
    ))
    .expect("parse samplelp");
    for solver in SOLVERS {
        let lp = LpProblem::from_matrix(
            parsed.matrix.as_inequality().expect("inequality matrix"),
            &eps,
        )
        .expect("build lp");
        let solution = lp.solve(solver, &eps);
        assert_eq!(
            solution.status(),
            LpStatus::Optimal,
            "{solver:?} status for samplelp"
        );
        assert!(
            solution.optimal_value().is_sign_positive(),
            "{solver:?} expected positive optimum, got {}",
            solution.optimal_value()
        );
    }
}

#[test]
fn records_pivot_counts() {
    let eps = f64::default_eps();
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplelp.ine",
    ))
    .expect("parse samplelp");
    for solver in SOLVERS {
        let lp = LpProblem::from_matrix(
            parsed.matrix.as_inequality().expect("inequality matrix"),
            &eps,
        )
        .expect("build lp");
        let solution = lp.solve(solver, &eps);
        assert_eq!(
            solution.status(),
            LpStatus::Optimal,
            "{solver:?} status for samplelp"
        );
        assert!(
            solution.pivots().total > 0,
            "{solver:?} expected at least one pivot to be recorded"
        );
    }
}

#[test]
fn criss_cross_solves_samplelp() {
    let eps = f64::default_eps();
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplelp.ine",
    ))
    .expect("parse samplelp");
    let lp = LpProblem::from_matrix(
        parsed.matrix.as_inequality().expect("inequality matrix"),
        &eps,
    )
    .expect("build lp");
    let solution = lp.solve(LpSolver::CrissCross, &eps);
    let lp_dual = LpProblem::from_matrix(
        parsed.matrix.as_inequality().expect("inequality matrix"),
        &eps,
    )
    .expect("build lp");
    let dual_solution = lp_dual.solve(LpSolver::DualSimplex, &eps);
    assert_eq!(solution.status(), LpStatus::Optimal);
    assert_eq!(dual_solution.status(), LpStatus::Optimal);
    assert!(
        approx_eq(
            *solution.optimal_value(),
            *dual_solution.optimal_value(),
            TestNumber::from(1.0e-9)
        ),
        "criss-cross optimum {} differs from dual simplex {}",
        solution.optimal_value(),
        dual_solution.optimal_value()
    );
}

#[test]
fn solving_is_pure() {
    let eps = f64::default_eps();
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplelp.ine",
    ))
    .expect("parse samplelp");
    for solver in SOLVERS {
        let lp = LpProblem::from_matrix(
            parsed.matrix.as_inequality().expect("inequality matrix"),
            &eps,
        )
        .expect("build lp");
        let first = lp.clone().solve(solver, &eps);
        let second = lp.solve(solver, &eps);
        assert_eq!(
            first.status(),
            second.status(),
            "{solver:?} should report the same status on repeated solves"
        );
        assert_eq!(
            first.optimal_value(),
            second.optimal_value(),
            "{solver:?} should produce a stable optimum"
        );
        assert_eq!(
            first.pivots().total,
            second.pivots().total,
            "{solver:?} pivot counts should be reproducible"
        );
    }
}

#[test]
fn pivot_limit_reports_exceeded_status() {
    let eps = f64::default_eps();
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplelp.ine",
    ))
    .expect("parse samplelp");
    for solver in SOLVERS {
        let baseline_lp = LpProblem::from_matrix(
            parsed.matrix.as_inequality().expect("inequality matrix"),
            &eps,
        )
        .expect("build lp");
        let baseline_solution = baseline_lp.solve(solver, &eps);
        assert_eq!(
            baseline_solution.status(),
            LpStatus::Optimal,
            "{solver:?} baseline"
        );
        let pivot_count = baseline_solution.pivots().phase_one
            + baseline_solution.pivots().phase_two
            + baseline_solution.pivots().criss_cross;
        assert!(pivot_count > 0, "{solver:?} expected pivots during solve");

        let capped_lp = LpProblem::from_matrix(
            parsed.matrix.as_inequality().expect("inequality matrix"),
            &eps,
        )
        .expect("build limited lp")
        .with_max_pivots(Some(0));
        let limited_solution = capped_lp.solve(solver, &eps);
        assert_eq!(
            limited_solution.status(),
            LpStatus::PivotLimitExceeded,
            "{solver:?} capped"
        );
        assert_ne!(limited_solution.status(), LpStatus::Inconsistent);
    }
}

#[test]
fn unconstrained_lp_is_unbounded() {
    let eps = f64::default_eps();
    let matrix = LpMatrixBuilder::<TestNumber, hullabaloo::types::Inequality>::with_columns(2)
        .with_row_vec(vec![0.0, 1.0])
        .with_objective(LpObjective::Maximize)
        .build();

    for solver in SOLVERS {
        let lp = LpProblem::from_matrix(&matrix, &eps).expect("build lp");
        let solution = lp.solve(solver, &eps);
        assert_eq!(
            solution.status(),
            LpStatus::Unbounded,
            "{solver:?} should report unbounded for an unconstrained objective"
        );
    }
}
