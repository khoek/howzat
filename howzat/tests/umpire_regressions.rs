#![cfg(feature = "rug")]

#[path = "support/common.rs"]
mod common;

use calculo::num::{DynamicEpsilon, F64Em12Epsilon, Num, RugRat};
use howzat::dd::ConeOptions;
use howzat::dd::{DefaultNormalizer, MultiPrecisionUmpire, SinglePrecisionUmpire, SnapPurifier};
use howzat::matrix::LpMatrix as Matrix;
use howzat::polyhedron::{DdConfig, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::types::{Generator, Inequality, Representation};
use std::path::Path;

fn rugrat_shadow_eps() -> DynamicEpsilon<RugRat> {
    DynamicEpsilon::new(RugRat::try_from_f64(1.0e-12).expect("default eps is finite"))
}

fn canonical_row_count_f64<R: Representation>(matrix: &Matrix<f64, R>) -> usize {
    matrix
        .normalized_sorted_unique(&F64Em12Epsilon)
        .0
        .row_count()
}

fn canonical_row_count_rug<R: Representation>(matrix: &Matrix<RugRat, R>) -> usize {
    let eps = DynamicEpsilon::new(RugRat::zero());
    matrix.normalized_sorted_unique(&eps).0.row_count()
}

fn to_rug_matrix(input: &Matrix<f64, Generator>) -> Matrix<RugRat, Generator> {
    let mut rows = Vec::with_capacity(input.row_count());
    for row in input.rows() {
        rows.push(
            row.iter()
                .map(|v| RugRat::try_from_f64(*v).expect("fixture entries must be finite"))
                .collect::<Vec<_>>(),
        );
    }
    Matrix::<RugRat, Generator>::from_rows(rows)
}

#[test]
fn d8_v18_seed5_single_precision_misses_facets_vs_exact() {
    let path = Path::new("tests/data/umpire_regressions/d8_v18_seed5.ext");
    let parsed = common::parse_cdd_file(path).expect("parse regression fixture");
    let input = match parsed.matrix {
        common::ParsedMatrix::Generator(m) => m,
        _ => panic!("expected generator input"),
    };

    let cone_options = ConeOptions::default();
    let poly_options = PolyhedronOptions::default();

    let single = PolyhedronOutput::<f64, Generator>::from_matrix_dd(
        input.clone(),
        DdConfig {
            cone: cone_options.clone(),
            poly: poly_options.clone(),
        },
        SinglePrecisionUmpire::new(F64Em12Epsilon),
    )
    .expect("single f64 dd");

    let purifying = PolyhedronOutput::<f64, Generator>::from_matrix_dd(
        input.clone(),
        DdConfig {
            cone: cone_options.clone(),
            poly: poly_options.clone(),
        },
        SinglePrecisionUmpire::with_purifier(
            F64Em12Epsilon,
            <f64 as DefaultNormalizer>::Norm::default(),
            SnapPurifier::new(),
        ),
    )
    .expect("purifying f64 dd");

    let multiprecision = PolyhedronOutput::<f64, Generator>::from_matrix_dd(
        input.clone(),
        DdConfig {
            cone: cone_options.clone(),
            poly: poly_options.clone(),
        },
        MultiPrecisionUmpire::<f64, RugRat, _, _>::new(F64Em12Epsilon, rugrat_shadow_eps()),
    )
    .expect("multiprecision f64 dd");

    let exact = PolyhedronOutput::<RugRat, Generator>::from_matrix_dd_int(
        to_rug_matrix(&input),
        DdConfig {
            cone: cone_options,
            poly: poly_options,
        },
    )
    .expect("exact RugRat dd");

    let single_h: &Matrix<f64, Inequality> = single.output_required();
    let purifying_h: &Matrix<f64, Inequality> = purifying.output_required();
    let multi_h: &Matrix<f64, Inequality> = multiprecision.output_required();
    let exact_h: &Matrix<RugRat, Inequality> = exact.output_required();

    let exact_facets = canonical_row_count_rug(exact_h);
    let single_facets = canonical_row_count_f64(single_h);
    let purifying_facets = canonical_row_count_f64(purifying_h);
    let multi_facets = canonical_row_count_f64(multi_h);

    eprintln!(
        "d8_v18_seed5 facet counts: single={single_facets} purifying={purifying_facets} multi={multi_facets} exact={exact_facets}"
    );

    assert!(
        single_facets < exact_facets,
        "single-precision unexpectedly matched exact facet count: single={single_facets} exact={exact_facets}"
    );

    assert!(
        exact_facets > 0,
        "exact run produced an empty H-rep unexpectedly"
    );

    // Purifying / MultiPrecision are exercised here but are not required to fix the issue yet.
    // A future improvement can upgrade these to match `exact_facets` for this fixture.
    let _ = (purifying_facets, multi_facets);
}

#[test]
#[ignore = "TODO: implement an upgrade/purification policy so MultiPrecisionUmpire matches the exact RugRat output"]
fn d8_v18_seed5_multiprecision_should_match_exact() {
    let path = Path::new("tests/data/umpire_regressions/d8_v18_seed5.ext");
    let parsed = common::parse_cdd_file(path).expect("parse regression fixture");
    let input = match parsed.matrix {
        common::ParsedMatrix::Generator(m) => m,
        _ => panic!("expected generator input"),
    };

    let cone_options = ConeOptions::default();
    let poly_options = PolyhedronOptions::default();

    let multiprecision = PolyhedronOutput::<f64, Generator>::from_matrix_dd(
        input.clone(),
        DdConfig {
            cone: cone_options.clone(),
            poly: poly_options.clone(),
        },
        MultiPrecisionUmpire::<f64, RugRat, _, _>::new(F64Em12Epsilon, rugrat_shadow_eps()),
    )
    .expect("multiprecision f64 dd");

    let exact = PolyhedronOutput::<RugRat, Generator>::from_matrix_dd_int(
        to_rug_matrix(&input),
        DdConfig {
            cone: cone_options,
            poly: poly_options,
        },
    )
    .expect("exact RugRat dd");

    let multi_h: &Matrix<f64, Inequality> = multiprecision.output_required();
    let exact_h: &Matrix<RugRat, Inequality> = exact.output_required();

    assert_eq!(
        canonical_row_count_f64(multi_h),
        canonical_row_count_rug(exact_h)
    );
}
