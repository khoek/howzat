#[path = "support/common.rs"]
mod common;

use calculo::num::Num;
use howzat::dd::ConeOptions;
use howzat::polyhedron::{Polyhedron, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::types::{IncidenceOutput, RepresentationKind};

use common::parse_cdd_file;

#[test]
fn converts_sample_h_representation() {
    let eps = f64::default_eps();
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/sampleh1.ine",
    ))
    .expect("parse sampleh1");
    let m = parsed
        .matrix
        .as_inequality()
        .expect("inequality matrix")
        .clone();
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

#[test]
fn converts_sample_v_representation() {
    let eps = f64::default_eps();
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplev1.ext",
    ))
    .expect("parse samplev1");
    let poly = PolyhedronOutput::<_, _>::builder(
        parsed
            .matrix
            .as_generator()
            .expect("generator matrix")
            .clone(),
    )
    .cone_options(ConeOptions::default())
    .run_dd_with_eps(eps)
    .expect("convert");
    let output = poly.output();
    assert_eq!(poly.representation(), RepresentationKind::Generator);
    assert_eq!(output.representation(), RepresentationKind::Inequality);
    assert!(output.row_count() > 0);
}

#[test]
fn redundant_rows_match_bug45_fixture() {
    let parsed = parse_cdd_file(&std::path::PathBuf::from("tests/data/examples/bug45.ine"))
        .expect("parse bug45");
    let result = parsed.matrix.redundant_rows();
    assert!(
        result.is_ok(),
        "redundant rows computation failed for bug45: {:?}",
        result.err()
    );
}
