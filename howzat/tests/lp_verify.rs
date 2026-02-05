#[path = "support/common.rs"]
mod common;

use calculo::num::{CoerceFrom, ConversionError, DynamicEpsilon, Num, RugRat};
use howzat::lp::LpProblem;
use howzat::lp::{LpObjective, LpSolver, LpStatus};
use howzat::matrix::LpMatrixBuilder;

use common::TestNumber;

const ZERO_EPS: f64 = 1.0e-9;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct FailingNumber(TestNumber);

impl std::ops::Add for FailingNumber {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for FailingNumber {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul for FailingNumber {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::ops::Div for FailingNumber {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl std::ops::Neg for FailingNumber {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Num for FailingNumber {
    type LinAlg = calculo::linalg::GenericOps;

    fn zero() -> Self {
        Self(0.0)
    }

    fn one() -> Self {
        Self(1.0)
    }

    fn from_u64(value: u64) -> Self {
        Self(value as f64)
    }

    fn ref_add(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    fn ref_sub(&self, other: &Self) -> Self {
        Self(self.0 - other.0)
    }

    fn ref_mul(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    fn ref_div(&self, other: &Self) -> Self {
        Self(self.0 / other.0)
    }

    fn ref_neg(&self) -> Self {
        Self(-self.0)
    }

    fn try_from_f64(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self(value))
    }

    fn to_f64(&self) -> f64 {
        self.0
    }

    fn hash_component(&self) -> u64 {
        const MANTISSA_MASK: u64 = (1u64 << 10) - 1;
        let mut bits = self.0.to_bits();
        if bits == (-0.0f64).to_bits() {
            bits = 0.0f64.to_bits();
        }
        bits & !MANTISSA_MASK
    }
}

impl CoerceFrom<TestNumber> for FailingNumber {
    fn coerce_from(_: &TestNumber) -> Result<Self, ConversionError> {
        Err(ConversionError)
    }
}

#[test]
fn verifies_solution_with_higher_precision_conversion() {
    let eps = TestNumber::default_eps();
    let matrix = LpMatrixBuilder::<TestNumber, hullabaloo::types::Inequality>::with_columns(3)
        .push_row(vec![1.0, -0.5, 0.0], false)
        .push_row(vec![1.0, 0.0, -0.5], false)
        .with_row_vec(vec![0.0, 1.0, 1.0])
        .with_objective(LpObjective::Maximize)
        .build();

    let lp = LpProblem::from_matrix(&matrix, &eps).expect("build lp");
    let solution = lp.solve_with_snapshot(LpSolver::DualSimplex, &eps);
    let eps_exact = RugRat::default_eps();
    let verification = solution.basis_status_as(&eps_exact);
    assert!(
        verification.is_valid(),
        "expected valid verification with RugRat converter, got {:?}",
        verification.issues()
    );
    assert!(
        !verification.conversion_failed(),
        "conversion should succeed"
    );
    assert!(verification.verification_pivots() > 0);
}

#[test]
fn conversion_failure_is_reported() {
    let eps = TestNumber::default_eps();
    let matrix = LpMatrixBuilder::<TestNumber, hullabaloo::types::Inequality>::from_rows(vec![
        vec![1.0, -1.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ])
    .with_row_vec(vec![0.0, 1.0, 1.0])
    .with_objective(LpObjective::Maximize)
    .build();

    let lp = LpProblem::from_matrix(&matrix, &eps).expect("build lp");
    let solution = lp.solve_with_snapshot(LpSolver::DualSimplex, &eps);
    let eps_fail = DynamicEpsilon::<FailingNumber>::new(FailingNumber(ZERO_EPS));
    let verification = solution.basis_status_as(&eps_fail);
    assert!(!verification.is_valid());
    assert!(verification.conversion_failed());
}

#[test]
fn snapshot_reuse_seeds_exact_solver() {
    let eps = TestNumber::default_eps();
    let matrix = LpMatrixBuilder::<TestNumber, hullabaloo::types::Inequality>::with_columns(3)
        .push_row(vec![1.0, -0.5, 0.0], false)
        .push_row(vec![1.0, 0.0, -0.5], false)
        .with_row_vec(vec![0.0, 1.0, 1.0])
        .with_objective(LpObjective::Maximize)
        .build();

    let float_lp = LpProblem::from_matrix(&matrix, &eps).expect("build float lp");
    let float_solution = float_lp.solve_with_snapshot(LpSolver::DualSimplex, &eps);
    let exact_snapshot = float_solution
        .snapshot()
        .expect("float snapshot")
        .coerce_as::<RugRat>()
        .expect("coerce snapshot");

    let mut builder =
        LpMatrixBuilder::<RugRat, hullabaloo::types::Inequality>::with_columns(matrix.col_count());
    for (idx, row) in matrix.rows().enumerate() {
        let converted: Vec<RugRat> = row
            .iter()
            .map(|v| RugRat::try_from_f64(*v).expect("convert coefficient"))
            .collect();
        builder = builder.push_row(converted, matrix.linearity().contains(idx));
    }
    let objective: Vec<RugRat> = matrix
        .row_vec()
        .iter()
        .map(|v| RugRat::try_from_f64(*v).expect("convert objective"))
        .collect();
    let exact_matrix = builder
        .with_row_vec(objective)
        .with_objective(matrix.objective())
        .build();

    let eps_exact = RugRat::default_eps();
    let exact_lp = LpProblem::from_matrix(&exact_matrix, &eps_exact).expect("build exact lp");
    let seeded_solution =
        exact_lp.solve_with_basis(LpSolver::DualSimplex, &exact_snapshot, &eps_exact);
    assert_eq!(seeded_solution.status(), LpStatus::Optimal);
    assert_eq!(
        seeded_solution
            .snapshot()
            .expect("exact snapshot")
            .nonbasic_index(),
        exact_snapshot.nonbasic_index()
    );
    let seeded_opt = seeded_solution.optimal_value().to_f64();
    let float_opt = float_solution.optimal_value().to_f64();
    assert!(
        (seeded_opt - float_opt).abs() < 1.0e-9,
        "seeded optimum {seeded_opt} should match float optimum {float_opt}"
    );
}

#[test]
fn solve_then_resolve_as_reuses_basis() {
    let eps = TestNumber::default_eps();
    let eps_exact = RugRat::default_eps();
    let matrix = LpMatrixBuilder::<TestNumber, hullabaloo::types::Inequality>::with_columns(3)
        .push_row(vec![1.0, -0.5, 0.0], false)
        .push_row(vec![1.0, 0.0, -0.5], false)
        .with_row_vec(vec![0.0, 1.0, 1.0])
        .with_objective(LpObjective::Maximize)
        .build();

    let lp = LpProblem::from_matrix(&matrix, &eps).expect("build lp");
    let (float_sol, exact_sol) = lp
        .solve_then_resolve_as::<RugRat>(LpSolver::DualSimplex, &eps, &eps_exact)
        .expect("solve twice");
    assert_eq!(float_sol.status(), LpStatus::Optimal);
    assert_eq!(exact_sol.status(), LpStatus::Optimal);
    let float_opt = float_sol.optimal_value().to_f64();
    let exact_opt = exact_sol.optimal_value().to_f64();
    assert!(
        (float_opt - exact_opt).abs() < 1.0e-9,
        "expected matching optima {float_opt} vs {exact_opt}"
    );
    assert_eq!(
        float_sol
            .snapshot()
            .expect("float snapshot")
            .nonbasic_index()
            .len(),
        exact_sol
            .snapshot()
            .expect("exact snapshot")
            .nonbasic_index()
            .len()
    );
}
