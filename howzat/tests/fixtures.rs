#[path = "support/common.rs"]
mod common;

use calculo::num::Num;
use howzat::dd::ConeOptions;
use howzat::matrix::LpMatrixBuilder;
use howzat::polyhedron::{Polyhedron, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::types::{ColSet, ComputationStatus, IncidenceOutput, RowSet};

use common::parse_cdd_file;

#[test]
fn redundant_and_adjacency_on_all_small_inputs() {
    let files = [
        "tests/data/examples/sampleh5.ine",
        "tests/data/examples/sampleh6.ine",
        "tests/data/examples/sampleh7.ine",
        "tests/data/examples/samplev3.ext",
        "tests/data/examples/redcheck.ext",
        "tests/data/examples-ine/infeas.ine",
    ];
    for path in files {
        let parsed = parse_cdd_file(&std::path::PathBuf::from(path)).expect("parse input");
        // Redundancy detection should not error.
        let red = parsed.matrix.redundant_rows();
        assert!(red.is_ok(), "redundant_rows failed for {}", path);
        // Canonicalization/adjacency paths exercised when feasible.
        let canon = parsed.matrix.canonicalize();
        assert!(canon.is_ok(), "canonicalize failed for {}", path);
        let adj = parsed.matrix.adjacency();
        assert!(adj.is_ok(), "adjacency failed for {}", path);
    }
}

#[test]
fn column_reduction_preserves_incidence_after_linearity() {
    let eps = f64::default_eps();
    let matrix =
        LpMatrixBuilder::from_rows(vec![vec![0.0, 1.0, 0.0], vec![0.0, -1.0, 0.0]]).build();

    let rank = matrix.storage().rank(
        &RowSet::new(matrix.row_count()),
        &ColSet::new(matrix.col_count()),
        &eps,
    );
    assert_eq!(rank.rank, 1, "expected rank 1 for the constructed matrix");

    let poly_options = PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        ..PolyhedronOptions::default()
    };
    let poly = Polyhedron::builder(matrix.clone())
        .cone_options(ConeOptions::default())
        .polyhedron_options(poly_options)
        .run_dd_with_eps(eps)
        .expect("build poly");

    assert_ne!(
        poly.status(),
        ComputationStatus::RegionEmpty,
        "polyhedron unexpectedly marked empty"
    );
    assert!(
        poly.linearity_dimension() > 0,
        "expected column reduction to detect linearity"
    );
    assert_eq!(
        poly.linearity_dimension(),
        1,
        "first column reduction should lower the reported linearity dimension"
    );

    let output = poly.output_required();
    assert_eq!(
        output.row_count(),
        2,
        "linearity rows should be reflected in the output"
    );

    let incidence = poly.incidence().expect("incidence");
    assert_eq!(incidence.family_size(), output.row_count());
    for set in incidence.sets() {
        assert_eq!(
            set.cardinality(),
            2,
            "each output row should be incident to both input inequalities"
        );
    }
}

#[test]
fn conversion_runs_on_diverse_inputs() {
    let eps = f64::default_eps();
    let files = [
        "tests/data/examples/sampleh1.ine",
        "tests/data/examples/sampleh2.ine",
        "tests/data/examples/sampleh3.ine",
        "tests/data/examples/samplev1.ext",
        "tests/data/examples/samplev2.ext",
        "tests/data/examples-ine/cube6.ine",
        "tests/data/examples-ine/cross6.ine",
    ];
    for path in files {
        let parsed = parse_cdd_file(&std::path::PathBuf::from(path)).expect("parse input");
        match parsed.matrix {
            common::ParsedMatrix::Inequality(m) => {
                assert!(
                    Polyhedron::builder(m)
                        .cone_options(ConeOptions::default())
                        .run_dd_with_eps(eps.clone())
                        .is_ok(),
                    "conversion failed for {}",
                    path
                );
            }
            common::ParsedMatrix::Generator(m) => {
                assert!(
                    PolyhedronOutput::<_, _>::builder(m)
                        .cone_options(ConeOptions::default())
                        .run_dd_with_eps(eps.clone())
                        .is_ok(),
                    "conversion failed for {}",
                    path
                );
            }
        };
    }
}

#[test]
fn relative_interior_for_lp_samples() {
    let eps = f64::default_eps();
    for name in [
        "samplelp.ine",
        "samplelp1.ine",
        "samplelp2.ine",
        "samplelp3.ine",
    ] {
        let parsed = parse_cdd_file(&std::path::PathBuf::from(format!(
            "tests/data/examples/{}",
            name
        )))
        .expect("parse lp");
        let interior = match parsed.matrix {
            common::ParsedMatrix::Inequality(m) => m.find_relative_interior(&eps),
            common::ParsedMatrix::Generator(m) => m.find_relative_interior(&eps),
        };
        assert!(interior.is_ok(), "relative interior failed for {}", name);
    }
}

#[test]
fn relative_interior_sets_strict_inequalities() {
    let eps = f64::default_eps();
    let matrix = LpMatrixBuilder::<f64, hullabaloo::types::Inequality>::with_columns(2)
        .push_row(vec![0.0, 1.0], false)
        .push_row(vec![1.0, -1.0], false)
        .build();

    let interior = matrix
        .find_relative_interior(&eps)
        .expect("relative interior");
    let solution = interior.lp_solution.expect("relative interior lp");
    let value = *solution.optimal_value();
    assert!(interior.exists, "expected non-empty relative interior");
    assert!(
        (value - 0.5).abs() < 1e-9,
        "expected max slack 0.5, got {}",
        value
    );
}
