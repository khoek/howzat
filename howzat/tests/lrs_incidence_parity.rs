#![cfg(feature = "rug")]

use calculo::num::{Num, RugRat};
use howzat::lrs::Options;
use howzat::matrix::{IncidenceConfig, LpMatrixBuilder};
use howzat::polyhedron::{LrsConfig, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{Generator, IncidenceOutput, Inequality, RowSet};

fn assert_set_family_rowwise_equal(label: &str, a: &SetFamily, b: &SetFamily) {
    assert_eq!(
        a.family_size(),
        b.family_size(),
        "{label}: family size mismatch"
    );
    assert_eq!(
        a.set_capacity(),
        b.set_capacity(),
        "{label}: set capacity mismatch"
    );

    for idx in 0..a.family_size() {
        let mut lhs: Vec<usize> = a
            .set(idx)
            .cloned()
            .unwrap_or_else(|| RowSet::new(a.set_capacity()))
            .iter()
            .map(|r| r.as_index())
            .collect();
        let mut rhs: Vec<usize> = b
            .set(idx)
            .cloned()
            .unwrap_or_else(|| RowSet::new(b.set_capacity()))
            .iter()
            .map(|r| r.as_index())
            .collect();
        lhs.sort_unstable();
        rhs.sort_unstable();
        assert_eq!(lhs, rhs, "{label}: set mismatch at row {idx}");
    }
}

#[test]
fn lrs_incidence_matches_dot_products_on_triangle_hrep() {
    // Triangle in R^2: x >= 0, y >= 0, x + y <= 1.
    let rat = |v: i32| RugRat::try_from_f64(v as f64).unwrap();
    let rows = vec![
        vec![rat(0), rat(1), rat(0)],   // x >= 0
        vec![rat(0), rat(0), rat(1)],   // y >= 0
        vec![rat(1), rat(-1), rat(-1)], // 1 - x - y >= 0
    ];
    let matrix = LpMatrixBuilder::<RugRat, Inequality>::from_rows(rows).build();

    let eps = RugRat::default_eps();
    let poly = PolyhedronOutput::<RugRat, Inequality>::from_matrix_lrs_as_exact::<RugRat, RugRat>(
        matrix,
        LrsConfig {
            poly: PolyhedronOptions {
                output_incidence: IncidenceOutput::Set,
                ..PolyhedronOptions::default()
            },
            lrs: Options::default(),
        },
        &eps,
    )
    .expect("LRS conversion");

    let incidence_lrs = poly.incidence().expect("LRS incidence");
    let incidence_dot = poly
        .input()
        .output_incidence_against(poly.output(), &eps, IncidenceConfig::default().no_slack())
        .expect("dot-product incidence");

    assert_set_family_rowwise_equal("triangle incidence", incidence_lrs, &incidence_dot);
}

#[test]
fn lrs_incidence_matches_dot_products_on_positive_orthant_cone_hrep() {
    // Cone in R^2: x >= 0, y >= 0 (homogeneous H-rep).
    let rat = |v: i32| RugRat::try_from_f64(v as f64).unwrap();
    let rows = vec![vec![rat(0), rat(1), rat(0)], vec![rat(0), rat(0), rat(1)]];
    let matrix = LpMatrixBuilder::<RugRat, Inequality>::from_rows(rows).build();

    let eps = RugRat::default_eps();
    let poly = PolyhedronOutput::<RugRat, Inequality>::from_matrix_lrs_as_exact::<RugRat, RugRat>(
        matrix,
        LrsConfig {
            poly: PolyhedronOptions {
                output_incidence: IncidenceOutput::Set,
                ..PolyhedronOptions::default()
            },
            lrs: Options::default(),
        },
        &eps,
    )
    .expect("LRS conversion");

    let incidence_lrs = poly.incidence().expect("LRS incidence");
    let incidence_dot = poly
        .input()
        .output_incidence_against(poly.output(), &eps, IncidenceConfig::default().no_slack())
        .expect("dot-product incidence");

    assert_set_family_rowwise_equal("orthant cone incidence", incidence_lrs, &incidence_dot);
}

#[test]
fn lrs_incidence_matches_dot_products_on_triangle_hull() {
    // Triangle in R^2: hull(points) -> inequalities.
    let rat = |v: i32| RugRat::try_from_f64(v as f64).unwrap();
    let points = [
        vec![rat(0), rat(0)],
        vec![rat(1), rat(0)],
        vec![rat(0), rat(1)],
    ];
    let rows: Vec<Vec<RugRat>> = points
        .iter()
        .map(|coords| {
            let mut row = Vec::with_capacity(coords.len() + 1);
            row.push(RugRat::one());
            row.extend_from_slice(coords);
            row
        })
        .collect();
    let matrix = LpMatrixBuilder::<RugRat, Generator>::from_rows(rows).build();

    let eps = RugRat::default_eps();
    let poly = PolyhedronOutput::<RugRat, Generator>::from_matrix_lrs_as_exact::<RugRat, RugRat>(
        matrix,
        LrsConfig {
            poly: PolyhedronOptions {
                output_incidence: IncidenceOutput::Set,
                ..PolyhedronOptions::default()
            },
            lrs: Options::default(),
        },
        &eps,
    )
    .expect("LRS conversion");

    let incidence_lrs = poly.incidence().expect("LRS incidence");
    let incidence_dot = poly
        .input()
        .output_incidence_against(poly.output(), &eps, IncidenceConfig::default().no_slack())
        .expect("dot-product incidence");

    assert_set_family_rowwise_equal("triangle hull incidence", incidence_lrs, &incidence_dot);
}

#[test]
fn lrs_incidence_matches_dot_products_on_cube_hull() {
    // Cube vertices in R^3: (x,y,z) in {0,1}^3.
    let rat = |v: i32| RugRat::try_from_f64(v as f64).unwrap();
    let mut rows = Vec::new();
    for x in 0..=1 {
        for y in 0..=1 {
            for z in 0..=1 {
                rows.push(vec![RugRat::one(), rat(x), rat(y), rat(z)]);
            }
        }
    }
    let matrix = LpMatrixBuilder::<RugRat, Generator>::from_rows(rows).build();

    let eps = RugRat::default_eps();
    let poly = PolyhedronOutput::<RugRat, Generator>::from_matrix_lrs_as_exact::<RugRat, RugRat>(
        matrix,
        LrsConfig {
            poly: PolyhedronOptions {
                output_incidence: IncidenceOutput::Set,
                ..PolyhedronOptions::default()
            },
            lrs: Options::default(),
        },
        &eps,
    )
    .expect("LRS conversion");

    let incidence_lrs = poly.incidence().expect("LRS incidence");
    let incidence_dot = poly
        .input()
        .output_incidence_against(poly.output(), &eps, IncidenceConfig::default().no_slack())
        .expect("dot-product incidence");

    assert_set_family_rowwise_equal("cube hull incidence", incidence_lrs, &incidence_dot);
}
