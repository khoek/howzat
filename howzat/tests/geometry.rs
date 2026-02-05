#[path = "support/common.rs"]
mod common;

use calculo::num::Num;
use howzat::dd::ConeOptions;
use howzat::polyhedron::{Polyhedron, PolyhedronOutput};

use common::parse_cdd_file;

#[test]
fn converts_three_dimensional_h_inputs() {
    let eps = f64::default_eps();
    for name in ["cube3.ine", "cubocta.ine", "dodeca.ine"] {
        let parsed = parse_cdd_file(&std::path::PathBuf::from(format!(
            "tests/data/examples-ine3d/{}",
            name
        )))
        .expect("parse 3d ine");
        let poly = Polyhedron::builder(
            parsed
                .matrix
                .as_inequality()
                .expect("inequality matrix")
                .clone(),
        )
        .cone_options(ConeOptions::default())
        .run_dd_with_eps(eps.clone())
        .expect("convert 3d ine");
        let out = poly.output();
        assert_eq!(
            out.representation(),
            hullabaloo::types::RepresentationKind::Generator
        );
    }
}

#[test]
fn converts_small_generator_examples() {
    let eps = f64::default_eps();
    for name in ["ccc4.ext", "ccp4.ext", "cyclic10-4.ext"] {
        let parsed = parse_cdd_file(&std::path::PathBuf::from(format!(
            "tests/data/examples-ext/{}",
            name
        )))
        .expect("parse ext");
        let poly = PolyhedronOutput::<_, _>::builder(
            parsed
                .matrix
                .as_generator()
                .expect("generator matrix")
                .clone(),
        )
        .cone_options(ConeOptions::default())
        .run_dd_with_eps(eps.clone())
        .expect("convert ext");
        let out = poly.output();
        assert_eq!(
            out.representation(),
            hullabaloo::types::RepresentationKind::Inequality
        );
    }
}
