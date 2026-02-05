#![cfg(feature = "rug")]

#[path = "support/common.rs"]
mod common;

use calculo::num::Num;
use howzat::dd::ConeOptions;
use howzat::lrs::Options;
use howzat::matrix::LpMatrixBuilder;
use howzat::polyhedron::{LrsConfig, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{AdjacencyOutput, Generator, Inequality, Representation, RowId};

use common::{normalize_matrix, parse_cdd_file};

fn assert_matrix_close<R: Representation>(
    a: &howzat::matrix::LpMatrix<f64, R>,
    b: &howzat::matrix::LpMatrix<f64, R>,
) {
    let tol = 1e-9;
    assert_eq!(a.row_count(), b.row_count(), "row count mismatch");
    assert_eq!(a.col_count(), b.col_count(), "col count mismatch");
    for i in 0..a.row_count() {
        let ra = a.row(i).expect("row in bounds");
        let rb = b.row(i).expect("row in bounds");
        for (va, vb) in ra.iter().zip(rb.iter()) {
            assert!(
                (va - vb).abs() <= tol,
                "entry mismatch at row {i}: left {va:?}, right {vb:?} (tol {tol})"
            );
        }
    }
    assert_eq!(a.linearity(), b.linearity(), "linearity set mismatch");
}

fn assert_matrix_exact_rug<R: Representation>(
    a: &howzat::matrix::LpMatrix<calculo::num::RugRat, R>,
    b: &howzat::matrix::LpMatrix<calculo::num::RugRat, R>,
) {
    assert_eq!(a.row_count(), b.row_count(), "row count mismatch");
    assert_eq!(a.col_count(), b.col_count(), "col count mismatch");
    assert_eq!(a.linearity(), b.linearity(), "linearity set mismatch");
    for i in 0..a.row_count() {
        let ra = a.row(i).expect("row in bounds");
        let rb = b.row(i).expect("row in bounds");
        for (va, vb) in ra.iter().zip(rb.iter()) {
            assert_eq!(
                va, vb,
                "entry mismatch at row {i}: left {va:?}, right {vb:?}"
            );
        }
    }
}

fn remap_set_family_by_positions(family: &SetFamily, positions: &[isize]) -> SetFamily {
    let new_size = positions.iter().filter(|p| **p >= 0).count();
    let mut builder = SetFamily::builder(new_size, new_size);

    let rep_pos = |idx: usize| -> usize {
        let p = positions[idx];
        if p >= 0 {
            return p as usize;
        }
        let rep_old = (-p - 1) as usize;
        let rep_p = positions[rep_old];
        assert!(rep_p >= 0, "duplicate row maps to non-representative");
        rep_p as usize
    };

    for (old_i, set) in family.sets().iter().enumerate() {
        if old_i >= positions.len() || positions[old_i] < 0 {
            continue;
        }
        let new_i = positions[old_i] as usize;
        for old_j in set.iter() {
            let j = old_j.as_index();
            if j >= positions.len() {
                continue;
            }
            let new_j = rep_pos(j);
            builder.insert_into_set(new_i, RowId::new(new_j));
        }
    }
    builder.build()
}

fn assert_set_family_equal_rowwise(label: &str, a: &SetFamily, b: &SetFamily) {
    assert_eq!(
        a.family_size(),
        b.family_size(),
        "{label}: family size mismatch"
    );
    let canon = |sf: &SetFamily| -> Vec<Vec<usize>> {
        sf.sets()
            .iter()
            .map(|s| {
                let mut elems: Vec<usize> = s.iter().map(|r| r.as_index()).collect();
                elems.sort_unstable();
                elems
            })
            .collect()
    };
    let ca = canon(a);
    let cb = canon(b);
    assert_eq!(ca, cb, "{label}: sets differed");
}

fn assert_outputs_match_dd_vs_lrs<R: hullabaloo::types::DualRepresentation>(
    label: &str,
    matrix: howzat::matrix::LpMatrix<f64, R>,
) {
    let eps = f64::default_eps();

    eprintln!("  [{label}] running DD…");
    let dd = PolyhedronOutput::<f64, R>::builder(matrix.clone())
        .cone_options(ConeOptions::default())
        .polyhedron_options(PolyhedronOptions::default())
        .run_dd_with_eps(eps.clone())
        .expect("DD conversion must succeed");
    eprintln!("  [{label}] DD done");

    eprintln!("  [{label}] running LRS…");
    let lrs = PolyhedronOutput::<f64, R>::from_matrix_lrs_as_exact::<calculo::num::RugRat, f64>(
        matrix,
        LrsConfig {
            poly: PolyhedronOptions::default(),
            lrs: Options::default(),
        },
        &eps,
    )
    .expect("LRS conversion must succeed");
    eprintln!("  [{label}] LRS done");

    assert_eq!(
        dd.status(),
        lrs.status(),
        "DD/LRS status mismatch for {label}"
    );

    eprintln!("  [{label}] normalizing + comparing outputs…");
    let dd_out = normalize_matrix(&dd.output_formatted(&eps));
    let lrs_out = normalize_matrix(&lrs.output_formatted(&eps));

    if dd_out.row_count() != lrs_out.row_count()
        || dd_out.col_count() != lrs_out.col_count()
        || dd_out.linearity() != lrs_out.linearity()
    {
        panic!(
            "DD/LRS outputs differed for {label}.\nDD:\n{}\nLRS:\n{}",
            common::format_matrix(&dd_out),
            common::format_matrix(&lrs_out),
        );
    }
    assert_matrix_close(&dd_out, &lrs_out);
    eprintln!("  [{label}] outputs match");
}

#[test]
fn lrs_matches_dd_on_sample_hrep() {
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/sampleh1.ine",
    ))
    .expect("parse sampleh1");
    let m = parsed
        .matrix
        .as_inequality()
        .expect("inequality matrix")
        .clone();

    assert_outputs_match_dd_vs_lrs::<Inequality>("tests/data/examples/sampleh1.ine", m);
}

#[test]
fn lrs_matches_dd_on_sample_hrep_with_output_adjacency() {
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/sampleh1.ine",
    ))
    .expect("parse sampleh1");
    let m = parsed
        .matrix
        .as_inequality()
        .expect("inequality matrix")
        .clone();

    let eps = f64::default_eps();
    let poly_options = PolyhedronOptions {
        output_adjacency: AdjacencyOutput::List,
        ..PolyhedronOptions::default()
    };
    let dd = PolyhedronOutput::<f64, Inequality>::builder(m.clone())
        .cone_options(ConeOptions::default())
        .polyhedron_options(poly_options.clone())
        .run_dd_with_eps(eps.clone())
        .expect("DD conversion must succeed");
    let lrs = PolyhedronOutput::<f64, Inequality>::from_matrix_lrs_as_exact::<
        calculo::num::RugRat,
        f64,
    >(
        m,
        LrsConfig {
            poly: poly_options,
            lrs: Options::default(),
        },
        &eps,
    )
    .expect("LRS conversion must succeed");

    assert_eq!(dd.status(), lrs.status(), "status mismatch");

    let (dd_out, dd_pos) = dd.output_formatted(&eps).normalized_sorted_unique(&eps);
    let (lrs_out, lrs_pos) = lrs.output_formatted(&eps).normalized_sorted_unique(&eps);
    assert_matrix_close(&dd_out, &lrs_out);

    let dd_adj =
        remap_set_family_by_positions(dd.adjacency().expect("DD output adjacency"), &dd_pos);
    let lrs_adj =
        remap_set_family_by_positions(lrs.adjacency().expect("LRS output adjacency"), &lrs_pos);
    assert_set_family_equal_rowwise("output adjacency", &dd_adj, &lrs_adj);
}

#[test]
fn dd_recompute_with_lrs_matches_direct_exact_lrs_on_sample_hrep() {
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/sampleh1.ine",
    ))
    .expect("parse sampleh1");
    let m = parsed
        .matrix
        .as_inequality()
        .expect("inequality matrix")
        .clone();

    let eps_f = f64::default_eps();
    let dd = PolyhedronOutput::<f64, Inequality>::builder(m.clone())
        .cone_options(ConeOptions::default())
        .polyhedron_options(PolyhedronOptions::default())
        .run_dd_with_eps(eps_f.clone())
        .expect("DD conversion must succeed");

    let eps_r = calculo::num::RugRat::default_eps();
    let completed = PolyhedronOutput::<f64, Inequality>::from_matrix_lrs_as_exact::<
        calculo::num::RugRat,
        calculo::num::RugRat,
    >(
        dd.input().clone(),
        LrsConfig {
            poly: PolyhedronOptions::default(),
            lrs: Options::default(),
        },
        &eps_r,
    )
    .expect("recompute must succeed");

    let direct = PolyhedronOutput::<f64, Inequality>::from_matrix_lrs_as_exact::<
        calculo::num::RugRat,
        calculo::num::RugRat,
    >(
        m,
        LrsConfig {
            poly: PolyhedronOptions::default(),
            lrs: Options::default(),
        },
        &eps_r,
    )
    .expect("direct LRS conversion must succeed");

    assert_eq!(completed.status(), direct.status(), "status mismatch");

    let completed_out = completed
        .output_formatted(&eps_r)
        .normalized_sorted_unique(&eps_r)
        .0;
    let direct_out = direct
        .output_formatted(&eps_r)
        .normalized_sorted_unique(&eps_r)
        .0;
    assert_matrix_exact_rug(&completed_out, &direct_out);
}

#[test]
fn dd_resolution_uses_incidence_certificate_when_available() {
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/sampleh1.ine",
    ))
    .expect("parse sampleh1");
    let m = parsed
        .matrix
        .as_inequality()
        .expect("inequality matrix")
        .clone();

    let eps_f = f64::default_eps();
    let dd = PolyhedronOutput::<f64, Inequality>::builder(m.clone())
        .cone_options(ConeOptions::default())
        .polyhedron_options(PolyhedronOptions {
            output_incidence: hullabaloo::types::IncidenceOutput::Set,
            ..PolyhedronOptions::default()
        })
        .run_dd_with_eps(eps_f.clone())
        .expect("DD conversion must succeed");
    dd.incidence().expect("DD must have incidence");

    let eps_r = calculo::num::RugRat::default_eps();
    let cert = dd.certificate().expect("DD must carry resolve certificate");
    let completed = cert
        .resolve_as::<calculo::num::RugRat>(
            PolyhedronOptions::default(),
            howzat::polyhedron::ResolveOptions::default(),
            &eps_r,
        )
        .expect("certificate resolution must succeed");

    let direct = PolyhedronOutput::<f64, Inequality>::from_matrix_lrs_as_exact::<
        calculo::num::RugRat,
        calculo::num::RugRat,
    >(
        m,
        LrsConfig {
            poly: PolyhedronOptions::default(),
            lrs: Options::default(),
        },
        &eps_r,
    )
    .expect("direct LRS conversion must succeed");

    assert_eq!(completed.status(), direct.status(), "status mismatch");
    let completed_out = completed
        .output_formatted(&eps_r)
        .normalized_sorted_unique(&eps_r)
        .0;
    let direct_out = direct
        .output_formatted(&eps_r)
        .normalized_sorted_unique(&eps_r)
        .0;
    assert_matrix_exact_rug(&completed_out, &direct_out);
}

#[test]
fn lrs_matches_dd_on_sample_vrep() {
    let parsed = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/samplev1.ext",
    ))
    .expect("parse samplev1");
    let m = parsed
        .matrix
        .as_generator()
        .expect("generator matrix")
        .clone();

    assert_outputs_match_dd_vs_lrs::<Generator>("tests/data/examples/samplev1.ext", m);
}

#[test]
fn lrs_matches_dd_on_triangle_hull() {
    // Triangle in R^2: hull(points) -> inequalities.
    let points = [vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let rows: Vec<Vec<f64>> = points
        .iter()
        .map(|coords| {
            let mut row = Vec::with_capacity(coords.len() + 1);
            row.push(1.0);
            row.extend_from_slice(coords);
            row
        })
        .collect();
    let matrix = LpMatrixBuilder::<f64, Generator>::from_rows(rows).build();

    assert_outputs_match_dd_vs_lrs::<Generator>("triangle hull", matrix);
}

#[test]
fn lrs_matches_dd_on_more_cdd_examples() {
    let examples = [
        // H-rep examples (includes explicit linearity + redundancy cases)
        "tests/data/examples/sampleh2.ine",
        "tests/data/examples/sampleh3.ine",
        "tests/data/examples/sampleh4.ine",
        "tests/data/examples/sampleh5.ine",
        "tests/data/examples/sampleh6.ine",
        "tests/data/examples/sampleh7.ine",
        // V-rep examples (includes rays + implicit linearity cases)
        "tests/data/examples/redcheck.ext",
        "tests/data/examples/samplev2.ext",
        "tests/data/examples/samplev3.ext",
    ];

    for path in examples {
        eprintln!("\n--- DD vs LRS parity: {path} ---");
        let parsed = parse_cdd_file(&std::path::PathBuf::from(path))
            .unwrap_or_else(|_| panic!("parse {path}"));
        match parsed.matrix {
            common::ParsedMatrix::Inequality(m) => {
                eprintln!("parsed as H-rep (inequality); running conversions…");
                assert_outputs_match_dd_vs_lrs::<Inequality>(path, m);
            }
            common::ParsedMatrix::Generator(m) => {
                eprintln!("parsed as V-rep (generator); running conversions…");
                assert_outputs_match_dd_vs_lrs::<Generator>(path, m);
            }
        }
        eprintln!("ok: {path}");
    }
}

#[test]
#[ignore = "slow: sampleh8 is a stress-test; DD baseline can take >60s locally"]
fn lrs_matches_dd_on_sampleh8_stress() {
    let path = "tests/data/examples/sampleh8.ine";
    eprintln!("\n--- DD vs LRS parity (slow): {path} ---");
    let parsed =
        parse_cdd_file(&std::path::PathBuf::from(path)).unwrap_or_else(|_| panic!("parse {path}"));
    let m = parsed
        .matrix
        .as_inequality()
        .expect("inequality matrix")
        .clone();
    assert_outputs_match_dd_vs_lrs::<Inequality>(path, m);
}
