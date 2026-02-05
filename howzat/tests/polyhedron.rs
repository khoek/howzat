#[path = "support/common.rs"]
mod common;

use calculo::num::Num;
use howzat::dd::ConeOptions;
use howzat::lp::LpObjective;
use howzat::polyhedron::{Polyhedron, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::types::{AdjacencyOutput, IncidenceOutput, RepresentationKind};

use common::parse_cdd_file;
use howzat::matrix::LpMatrixBuilder;

#[test]
fn polyhedron_adjacency_and_incidence_exist_when_conversion_succeeds() {
    let eps = f64::default_eps();
    for path in [
        "tests/data/examples/sampleh2.ine",
        "tests/data/examples/samplev2.ext",
    ] {
        let parsed = parse_cdd_file(&std::path::PathBuf::from(path)).expect("parse input");
        let poly_options = PolyhedronOptions {
            output_incidence: IncidenceOutput::Set,
            input_incidence: IncidenceOutput::Off,
            output_adjacency: AdjacencyOutput::List,
            input_adjacency: AdjacencyOutput::Off,
            save_basis_and_tableau: false,
            save_repair_hints: false,
            profile_adjacency: false,
        };
        match parsed.matrix {
            common::ParsedMatrix::Inequality(m) => {
                let poly = Polyhedron::builder(m)
                    .cone_options(ConeOptions::default())
                    .polyhedron_options(poly_options.clone())
                    .run_dd_with_eps(eps.clone())
                    .expect("convert");
                let out = poly.output();
                assert_eq!(out.representation(), RepresentationKind::Generator);
                assert!(
                    poly.incidence().is_some(),
                    "expected incidence to be present"
                );
                assert!(
                    poly.adjacency().is_some(),
                    "expected adjacency to be present"
                );
            }
            common::ParsedMatrix::Generator(m) => {
                let poly = PolyhedronOutput::<_, _>::builder(m)
                    .cone_options(ConeOptions::default())
                    .polyhedron_options(poly_options)
                    .run_dd_with_eps(eps.clone())
                    .expect("convert");
                let out = poly.output();
                assert_eq!(out.representation(), RepresentationKind::Inequality);
                assert!(
                    poly.incidence().is_some(),
                    "expected incidence to be present"
                );
                assert!(
                    poly.adjacency().is_some(),
                    "expected adjacency to be present"
                );
            }
        }
    }
}

#[test]
fn output_matrix_preserves_objective_metadata() {
    let eps = f64::default_eps();
    let mut parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplev1.ext",
    ))
    .expect("parse generator input");
    let matrix = parsed.matrix.as_generator().expect("generator matrix");
    let modified_matrix = LpMatrixBuilder::from_matrix(matrix)
        .with_rows(matrix.rows().map(|r| r.to_vec()).collect())
        .with_linearity(matrix.linearity().clone())
        .with_row_vec(vec![1.0, 2.0, 3.0, 4.0])
        .with_objective(LpObjective::Maximize)
        .build();

    let poly = PolyhedronOutput::<_, _>::builder(modified_matrix.clone())
        .cone_options(ConeOptions::default())
        .run_dd_with_eps(eps)
        .expect("convert");
    let output = poly.output();

    // We need to update parsed.matrix to reflect the modified_matrix for the assertions to pass
    // as the original assertions compare against parsed.matrix.
    parsed.matrix = common::ParsedMatrix::Generator(modified_matrix);

    assert_eq!(output.objective(), parsed.matrix.objective());
    assert_eq!(output.row_vec(), parsed.matrix.row_vec());
}
