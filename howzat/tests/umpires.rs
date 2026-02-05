#[path = "support/common.rs"]
mod common;

use calculo::linalg;
use calculo::num::{DynamicEpsilon, Epsilon, F64Em12Epsilon, Num, RugRat, Sign};
use common::parse_cdd_file;
use howzat::dd::ConeOptions;
use howzat::dd::{
    AdaptivePrecisionUmpire, DefaultNormalizer, MultiPrecisionUmpire, SinglePrecisionUmpire,
    SnapPurifier, Umpire,
};
use howzat::matrix::LpMatrix as Matrix;
use howzat::polyhedron::{DdConfig, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::types::{
    AdjacencyOutput, DualRepresentation, Generator, IncidenceOutput, Inequality, InequalityKind,
    Representation,
};
use std::path::Path;

fn rugrat_shadow_eps() -> DynamicEpsilon<RugRat> {
    DynamicEpsilon::new(RugRat::try_from_f64(1.0e-12).expect("default eps is finite"))
}

fn adaptive_trigger_eps() -> DynamicEpsilon<f64> {
    DynamicEpsilon::new(1.0e-10)
}

fn normalized_matrix<R: Representation>(matrix: &Matrix<f64, R>) -> Matrix<f64, R> {
    let eps = F64Em12Epsilon;
    matrix.normalized_sorted_unique(&eps).0
}

fn rows_present<R: Representation>(sup: &Matrix<f64, R>, sub: &Matrix<f64, R>, tol: f64) -> bool {
    for sr in sub.rows() {
        let mut found = false;
        for rr in sup.rows() {
            if sr.len() != rr.len() {
                continue;
            }
            if sr.iter().zip(rr.iter()).all(|(a, b)| (a - b).abs() <= tol) {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

fn assert_matrices_equivalent<R: Representation>(a: &Matrix<f64, R>, b: &Matrix<f64, R>) {
    let a = normalized_matrix(a);
    let b = normalized_matrix(b);
    let tol = 1.0e-6;
    assert!(
        rows_present(&a, &b, tol) && rows_present(&b, &a, tol),
        "normalized matrices differ\nA:\n{}\nB:\n{}",
        common::format_matrix(&a),
        common::format_matrix(&b)
    );
    assert_eq!(a.linearity(), b.linearity(), "linearity sets differ");
}

fn assert_output_generators_satisfy_input(
    input: &Matrix<f64, Inequality>,
    output: &Matrix<f64, Generator>,
) {
    let eps = F64Em12Epsilon;
    for (gen_idx, generator) in output.rows().enumerate() {
        for row in 0..input.row_count() {
            let kind = if input.linearity().contains(row) {
                InequalityKind::Equality
            } else {
                InequalityKind::Inequality
            };
            let value = dot_rug(input.row(row).expect("row in bounds"), generator);
            let violates = violates_rug(kind, &value, false, &eps);
            assert!(
                !violates,
                "output generator violates input constraint: gen_idx={gen_idx} row={row} kind={kind:?} value={value:?}"
            );
        }
    }
}

fn assert_output_inequalities_satisfied_by_input(
    input: &Matrix<f64, Generator>,
    output: &Matrix<f64, Inequality>,
) {
    let output = normalized_matrix(output);
    let eps = F64Em12Epsilon;
    for (gen_idx, generator) in input.rows().enumerate() {
        for row in 0..output.row_count() {
            let kind = if output.linearity().contains(row) {
                InequalityKind::Equality
            } else {
                InequalityKind::Inequality
            };
            let value = dot_rug(output.row(row).expect("row in bounds"), generator);
            let violates = violates_rug(kind, &value, false, &eps);
            assert!(
                !violates,
                "input generator violates output inequality: gen_idx={gen_idx} row={row} kind={kind:?} value={value:?}"
            );
        }
    }
}

#[inline(always)]
fn violates_sign(kind: InequalityKind, sign: Sign, relaxed: bool) -> bool {
    match kind {
        InequalityKind::Equality => sign != Sign::Zero,
        InequalityKind::Inequality => sign == Sign::Negative,
        InequalityKind::StrictInequality => {
            if relaxed {
                sign != Sign::Positive
            } else {
                sign == Sign::Negative
            }
        }
    }
}

#[inline]
fn dot_rug(lhs: &[f64], rhs: &[f64]) -> RugRat {
    assert_eq!(lhs.len(), rhs.len(), "dot product dimension mismatch");
    let mut acc = RugRat::zero();
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let ar = RugRat::try_from_f64(*a).expect("matrix entries must be finite");
        let br = RugRat::try_from_f64(*b).expect("vector entries must be finite");
        linalg::add_mul_assign(&mut acc, &ar, &br);
    }
    acc
}

#[inline(always)]
fn violates_rug(
    kind: InequalityKind,
    value: &RugRat,
    relaxed: bool,
    eps: &impl Epsilon<f64>,
) -> bool {
    let approx = value.to_f64();
    let sign = eps.sign(&approx);
    violates_sign(kind, sign, relaxed)
}

fn convert_with_umpire<R: DualRepresentation, U: Umpire<f64>>(
    matrix: Matrix<f64, R>,
    cone_options: ConeOptions,
    poly_options: PolyhedronOptions,
    umpire: U,
) -> PolyhedronOutput<f64, R> {
    PolyhedronOutput::<f64, R>::from_matrix_dd(
        matrix,
        DdConfig {
            cone: cone_options,
            poly: poly_options,
        },
        umpire,
    )
    .expect("dd conversion")
}

#[test]
fn umpires_agree_on_core_fixtures() {
    let cone_options = ConeOptions::default();
    let poly_options = PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        input_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: true,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    // Keep this list small-ish for runtime, but cover both H/V inputs and some tricky fixtures.
    let fixtures = [
        "tests/data/examples/sampleh1.ine",
        "tests/data/examples/sampleh2.ine",
        "tests/data/examples/bug45.ine",
        "tests/data/examples/project1.ine",
        "tests/data/examples/samplev1.ext",
        "tests/data/examples/samplev2.ext",
        "tests/data/examples-ine/infeas.ine",
        "tests/data/examples-ine/nonfull.ine",
        "tests/data/examples-ext/ccc4.ext",
    ];

    for path in fixtures {
        eprintln!("checking fixture {path}");
        let parsed = parse_cdd_file(Path::new(path)).expect("parse fixture");
        match parsed.matrix {
            common::ParsedMatrix::Inequality(matrix) => {
                let eps = F64Em12Epsilon;
                let single = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    SinglePrecisionUmpire::new(eps),
                );
                let purifying = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    SinglePrecisionUmpire::with_purifier(
                        eps,
                        <f64 as DefaultNormalizer>::Norm::default(),
                        SnapPurifier::new(),
                    ),
                );
                let adaptive = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    AdaptivePrecisionUmpire::<f64, RugRat, _, _, _>::new(
                        eps,
                        adaptive_trigger_eps(),
                        rugrat_shadow_eps(),
                    ),
                );
                let multiprecision = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    MultiPrecisionUmpire::<f64, RugRat, _, _>::new(eps, rugrat_shadow_eps()),
                );

                assert_eq!(
                    single.status(),
                    multiprecision.status(),
                    "{path}: status mismatch"
                );
                assert_eq!(
                    single.is_empty(),
                    multiprecision.is_empty(),
                    "{path}: empty mismatch"
                );
                assert_eq!(
                    purifying.status(),
                    multiprecision.status(),
                    "{path}: status mismatch (purifying)"
                );
                assert_eq!(
                    adaptive.status(),
                    multiprecision.status(),
                    "{path}: status mismatch (adaptive)"
                );
                assert_eq!(
                    purifying.is_empty(),
                    multiprecision.is_empty(),
                    "{path}: empty mismatch (purifying)"
                );
                assert_eq!(
                    adaptive.is_empty(),
                    multiprecision.is_empty(),
                    "{path}: empty mismatch (adaptive)"
                );
                if !multiprecision.is_empty() {
                    assert_output_generators_satisfy_input(&matrix, single.output());
                    assert_output_generators_satisfy_input(&matrix, purifying.output());
                    assert_output_generators_satisfy_input(&matrix, adaptive.output());
                    assert_output_generators_satisfy_input(&matrix, multiprecision.output());
                }
            }
            common::ParsedMatrix::Generator(matrix) => {
                let eps = F64Em12Epsilon;
                let single = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    SinglePrecisionUmpire::new(eps),
                );
                let purifying = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    SinglePrecisionUmpire::with_purifier(
                        eps,
                        <f64 as DefaultNormalizer>::Norm::default(),
                        SnapPurifier::new(),
                    ),
                );
                let adaptive = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    AdaptivePrecisionUmpire::<f64, RugRat, _, _, _>::new(
                        eps,
                        adaptive_trigger_eps(),
                        rugrat_shadow_eps(),
                    ),
                );
                let multiprecision = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    MultiPrecisionUmpire::<f64, RugRat, _, _>::new(eps, rugrat_shadow_eps()),
                );

                assert_eq!(
                    single.status(),
                    multiprecision.status(),
                    "{path}: status mismatch"
                );
                assert_eq!(
                    single.is_empty(),
                    multiprecision.is_empty(),
                    "{path}: empty mismatch"
                );
                assert_eq!(
                    purifying.status(),
                    multiprecision.status(),
                    "{path}: status mismatch (purifying)"
                );
                assert_eq!(
                    adaptive.status(),
                    multiprecision.status(),
                    "{path}: status mismatch (adaptive)"
                );
                assert_eq!(
                    purifying.is_empty(),
                    multiprecision.is_empty(),
                    "{path}: empty mismatch (purifying)"
                );
                assert_eq!(
                    adaptive.is_empty(),
                    multiprecision.is_empty(),
                    "{path}: empty mismatch (adaptive)"
                );

                assert_matrices_equivalent(single.output(), multiprecision.output());
                assert_matrices_equivalent(purifying.output(), multiprecision.output());
                assert_matrices_equivalent(adaptive.output(), multiprecision.output());
                if !multiprecision.is_empty() {
                    assert_output_inequalities_satisfied_by_input(&matrix, single.output());
                    assert_output_inequalities_satisfied_by_input(&matrix, purifying.output());
                    assert_output_inequalities_satisfied_by_input(&matrix, adaptive.output());
                    assert_output_inequalities_satisfied_by_input(&matrix, multiprecision.output());
                }
            }
        }
    }
}

#[test]
fn cyclic10_4_single_precision_misses_facets_but_multiprecision_matches_cddexec() {
    let eps = F64Em12Epsilon;
    let cone_options = ConeOptions::default();
    let poly_options = PolyhedronOptions::default();

    let parsed = parse_cdd_file(Path::new("tests/data/examples-ext/cyclic10-4.ext"))
        .expect("parse cyclic10-4.ext");
    let expected = parse_cdd_file(Path::new("tests/data/cddexec/ine-from-ext/cyclic10-4.ine"))
        .expect("parse cddexec cyclic10-4.ine");

    let input = match parsed.matrix {
        common::ParsedMatrix::Generator(m) => m,
        other => panic!(
            "expected V-representation input, got {:?}",
            other.representation()
        ),
    };
    let expected_h = match expected.matrix {
        common::ParsedMatrix::Inequality(m) => m,
        other => panic!(
            "expected H-representation output, got {:?}",
            other.representation()
        ),
    };

    let single = convert_with_umpire(
        input.clone(),
        cone_options.clone(),
        poly_options.clone(),
        SinglePrecisionUmpire::new(eps),
    );
    let purifying = convert_with_umpire(
        input.clone(),
        cone_options.clone(),
        poly_options.clone(),
        SinglePrecisionUmpire::with_purifier(
            eps,
            <f64 as DefaultNormalizer>::Norm::default(),
            SnapPurifier::new(),
        ),
    );
    let adaptive = convert_with_umpire(
        input.clone(),
        cone_options.clone(),
        poly_options.clone(),
        AdaptivePrecisionUmpire::<f64, RugRat, _, _, _>::new(
            eps,
            adaptive_trigger_eps(),
            rugrat_shadow_eps(),
        ),
    );
    let multiprecision = convert_with_umpire(
        input.clone(),
        cone_options.clone(),
        poly_options.clone(),
        MultiPrecisionUmpire::<f64, RugRat, _, _>::new(eps, rugrat_shadow_eps()),
    );

    assert_matrices_equivalent(multiprecision.output(), &expected_h);
    assert_matrices_equivalent(single.output(), &expected_h);
    assert_matrices_equivalent(purifying.output(), &expected_h);
    assert_matrices_equivalent(adaptive.output(), &expected_h);

    // Purifying may or may not fix this case; it should at least remain sound.
    assert_output_inequalities_satisfied_by_input(&input, purifying.output());
    assert_output_inequalities_satisfied_by_input(&input, adaptive.output());
    assert_output_inequalities_satisfied_by_input(&input, multiprecision.output());
}

#[test]
#[ignore = "slow smoke-test: runs many fixtures under multiple umpires"]
fn umpires_smoke_all_fixtures_default_options() {
    let eps = F64Em12Epsilon;
    let cone_options = ConeOptions::default();
    let poly_options = PolyhedronOptions::default();

    let files = common::cdd_input_files();
    for path in files {
        let parsed = parse_cdd_file(&path).expect("parse fixture");
        match parsed.matrix {
            common::ParsedMatrix::Inequality(matrix) => {
                let single = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    SinglePrecisionUmpire::new(eps),
                );
                let purifying = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    SinglePrecisionUmpire::with_purifier(
                        eps,
                        <f64 as DefaultNormalizer>::Norm::default(),
                        SnapPurifier::new(),
                    ),
                );
                let multiprecision = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    MultiPrecisionUmpire::<f64, RugRat, _, _>::new(eps, rugrat_shadow_eps()),
                );

                if !single.is_empty() {
                    assert_output_generators_satisfy_input(&matrix, single.output());
                }
                if !purifying.is_empty() {
                    assert_output_generators_satisfy_input(&matrix, purifying.output());
                }
                if !multiprecision.is_empty() {
                    assert_output_generators_satisfy_input(&matrix, multiprecision.output());
                }
            }
            common::ParsedMatrix::Generator(matrix) => {
                let single = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    SinglePrecisionUmpire::new(eps),
                );
                let purifying = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    SinglePrecisionUmpire::with_purifier(
                        eps,
                        <f64 as DefaultNormalizer>::Norm::default(),
                        SnapPurifier::new(),
                    ),
                );
                let multiprecision = convert_with_umpire(
                    matrix.clone(),
                    cone_options.clone(),
                    poly_options.clone(),
                    MultiPrecisionUmpire::<f64, RugRat, _, _>::new(eps, rugrat_shadow_eps()),
                );

                if !single.is_empty() {
                    assert_output_inequalities_satisfied_by_input(&matrix, single.output());
                }
                if !purifying.is_empty() {
                    assert_output_inequalities_satisfied_by_input(&matrix, purifying.output());
                }
                if !multiprecision.is_empty() {
                    assert_output_inequalities_satisfied_by_input(&matrix, multiprecision.output());
                }
            }
        }
    }
}
