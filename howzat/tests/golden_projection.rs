#[path = "support/common.rs"]
mod common;

use calculo::num::Num;
use common::{format_matrix, normalize_matrix, parse_cdd_file, rows_present};
use howzat::dd::ConeOptions;
use howzat::polyhedron::{Polyhedron, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::types::{IncidenceOutput, RepresentationKind};

fn run_fourier_projection_ex(name: &str) {
    let eps = f64::default_eps();
    let input = parse_cdd_file(&std::path::PathBuf::from(format!(
        "tests/data/examples/{}.ine",
        name
    )))
    .unwrap();
    let expected = parse_cdd_file(&std::path::PathBuf::from(format!(
        "tests/data/examples/{}res.ine",
        name
    )))
    .unwrap();
    let spec = input
        .project
        .as_ref()
        .expect("projection metadata should exist");

    match input.matrix.clone() {
        common::ParsedMatrix::Inequality(m) => {
            let poly = Polyhedron::builder(m)
                .cone_options(ConeOptions::default())
                .polyhedron_options(PolyhedronOptions {
                    output_incidence: IncidenceOutput::Set,
                    ..PolyhedronOptions::default()
                })
                .run_dd_with_eps(eps)
                .expect("convert");
            let out = poly.output();
            assert_eq!(out.representation(), RepresentationKind::Generator);
            assert!(
                poly.incidence().is_some(),
                "expected incidence to be present"
            );
        }
        common::ParsedMatrix::Generator(m) => {
            let poly = PolyhedronOutput::<_, _>::builder(m)
                .cone_options(ConeOptions::default())
                .run_dd_with_eps(eps)
                .expect("convert");
            let out = poly.output();
            assert_eq!(out.representation(), RepresentationKind::Inequality);
        }
    };

    let mut keep_cols = hullabaloo::types::ColSet::new(input.matrix.col_count());
    for col in &spec.keep_columns {
        keep_cols.insert(*col);
    }
    let projected = input
        .matrix
        .fourier_project(spec.dimension, &keep_cols)
        .expect("fourier projection");
    let norm_out = normalize_matrix(projected.as_inequality().expect("inequality matrix"));
    let norm_expected =
        normalize_matrix(expected.matrix.as_inequality().expect("inequality matrix"));
    assert!(
        rows_present(&norm_out, &norm_expected, 1e-6),
        "projection {name} mismatch\ncomputed:\n{}\nexpected:\n{}",
        format_matrix(&norm_out),
        format_matrix(&norm_expected)
    );
}

#[test]
fn fourier_projection_matches_project1() {
    run_fourier_projection_ex("project1");
}

#[test]
#[ignore = "Slow projection examples; skip in default test runs"]
fn fourier_projection_matches_project2() {
    run_fourier_projection_ex("project2");
}
