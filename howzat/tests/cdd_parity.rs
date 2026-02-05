use anyhow::Result;
use calculo::num::Num;
use cddlib_rs::{Matrix as CddMatrix, NumberType, Polyhedron as CddPolyhedron};
use howzat::dd::ConeOptions;
use howzat::matrix::LpMatrixBuilder;
use howzat::polyhedron::{PolyhedronOptions, PolyhedronOutput};
use hullabaloo::types::{Generator, IncidenceOutput};

fn build_generator_matrix(points: &[Vec<f64>]) -> Result<howzat::matrix::LpMatrix<f64, Generator>> {
    let rows: Vec<Vec<f64>> = points
        .iter()
        .map(|coords| {
            let mut row = Vec::with_capacity(coords.len() + 1);
            row.push(1.0);
            row.extend_from_slice(coords);
            row
        })
        .collect();
    Ok(LpMatrixBuilder::from_rows(rows).build())
}

fn cdd_incidence(points: &[Vec<f64>]) -> Result<Vec<Vec<usize>>> {
    let mut cdd: CddMatrix<f64, Generator> = CddMatrix::new(
        points.len(),
        points.first().map_or(0, |p| p.len() + 1),
        NumberType::Real,
    )?;
    for (r, coords) in points.iter().enumerate() {
        cdd.set_real(r, 0, 1.0);
        for (c, &val) in coords.iter().enumerate() {
            cdd.set_real(r, c + 1, val);
        }
        cdd.set_generator_type(r, true);
    }
    let poly = CddPolyhedron::from_matrix(&cdd)?;
    let mut facets = poly.incidence()?.to_adjacency_lists();
    for face in facets.iter_mut() {
        face.sort_unstable();
    }
    facets.sort();
    Ok(facets)
}

fn howzat_incidence(points: &[Vec<f64>]) -> Result<Vec<Vec<usize>>> {
    let matrix = build_generator_matrix(points)?;
    let eps = f64::default_eps();
    let poly = PolyhedronOutput::<f64, Generator>::builder(matrix)
        .cone_options(ConeOptions::default())
        .polyhedron_options(PolyhedronOptions {
            output_incidence: IncidenceOutput::Set,
            ..PolyhedronOptions::default()
        })
        .run_dd_with_eps(eps)?;
    let incidence = poly.incidence_required();
    let linearity = poly.output().linearity().clone();
    let mut faces = Vec::new();
    for idx in 0..incidence.family_size() {
        if linearity.contains(idx) {
            continue;
        }
        let set = incidence
            .set(idx)
            .cloned()
            .unwrap_or_else(|| hullabaloo::types::RowSet::new(points.len()));
        let mut verts: Vec<usize> = set.iter().map(|v| v.as_index()).collect();
        verts.sort_unstable();
        faces.push(verts);
    }
    faces.sort();
    Ok(faces)
}

fn verify_against_cdd(points: &[Vec<f64>]) -> Result<()> {
    let expected = cdd_incidence(points)?;
    let actual = howzat_incidence(points)?;
    assert_eq!(
        actual, expected,
        "howzat incidence differed from cddlib for points {:?}",
        points
    );
    Ok(())
}

#[test]
fn triangle_matches_cdd() -> Result<()> {
    let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    verify_against_cdd(&points)
}

#[test]
fn triangular_prism_matches_cdd() -> Result<()> {
    let points = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
    ];
    verify_against_cdd(&points)
}

#[test]
fn cube_matches_cdd() -> Result<()> {
    let mut points = Vec::new();
    for &x in &[0.0, 1.0] {
        for &y in &[0.0, 1.0] {
            for &z in &[0.0, 1.0] {
                points.push(vec![x, y, z]);
            }
        }
    }
    verify_against_cdd(&points)
}

#[test]
fn simplex_4d_matches_cdd() -> Result<()> {
    let points = vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    verify_against_cdd(&points)
}

#[test]
fn simplex_5d_matches_cdd() -> Result<()> {
    let points = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    verify_against_cdd(&points)
}
