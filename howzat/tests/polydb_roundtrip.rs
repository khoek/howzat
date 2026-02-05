#[path = "support/common.rs"]
mod common;
#[path = "support/polydb.rs"]
mod polydb;

use anyhow::{Context, Result, anyhow};
use calculo::num::Num;
use cddlib_rs::{Matrix as CddMatrix, NumberType, Polyhedron as CddPolyhedron};
use howzat::dd::ConeOptions;
use howzat::matrix::LpMatrix as Matrix;
use howzat::polyhedron::{Polyhedron, PolyhedronOutput};
use hullabaloo::types::{Generator, Inequality};

use polydb::{
    ItemSummary, ParsedMatrix, Stats, affine_rank, approx_eq, fetch_random_polytopes,
    item_to_matrix,
};

const SAMPLE_LIMIT: usize = 128;

#[test]
#[ignore = "PolyDB roundtrip depends on remote data; skipping during local iterations"]
fn polydb_stats_roundtrip() -> Result<()> {
    let eps = f64::default_eps();
    let sample = fetch_random_polytopes(SAMPLE_LIMIT).context("fetching PolyDB sample")?;
    assert!(
        !sample.items.is_empty(),
        "PolyDB returned no polytopes to validate"
    );

    let mut validated = 0usize;
    for item in sample.items.iter() {
        let label = item.id.as_deref().unwrap_or("<polydb entry without _id>");
        if item.definition.is_none() {
            continue;
        }
        let matrix = item_to_matrix(item).with_context(|| {
            format!(
                "failed to build matrix for {}",
                item.id.as_deref().unwrap_or("<polydb entry without _id>")
            )
        })?;
        let expected = cdd_stats_from_matrix(&matrix, label)?;
        let computed = match &matrix {
            ParsedMatrix::Inequality(m) => {
                let poly = Polyhedron::builder(m.clone())
                    .cone_options(ConeOptions::default())
                    .run_dd_with_eps(eps.clone())
                    .map_err(|e| {
                        anyhow!("dd conversion failed for {:?}: {e:?}", item.id.as_deref())
                    })?;
                let mut stats = hrep_stats(m);
                let gens = poly.output();
                merge_stats(&mut stats, generator_stats(gens));
                stats
            }
            ParsedMatrix::Generator(m) => {
                let poly = PolyhedronOutput::<_, _>::builder(m.clone())
                    .cone_options(ConeOptions::default())
                    .run_dd_with_eps(eps.clone())
                    .map_err(|e| {
                        anyhow!("dd conversion failed for {:?}: {e:?}", item.id.as_deref())
                    })?;
                let mut stats = generator_stats(m);
                let hrep = poly.output();
                merge_stats(&mut stats, hrep_stats(hrep));
                stats
            }
        };
        assert_stats_match(&expected, &computed, item);
        validated += 1;
    }
    assert!(
        validated > 0,
        "no PolyDB entries contained usable definitions"
    );

    Ok(())
}

fn generator_stats(matrix: &Matrix<f64, Generator>) -> Stats {
    let mut stats = Stats::default();
    let mut vertices = Vec::new();
    let mut ray_count = 0usize;
    let line_count = matrix.linearity().cardinality();

    for (idx, row) in matrix.rows().enumerate() {
        if matrix.linearity().contains(idx + 1) {
            continue;
        }
        if approx_eq(row[0], 0.0) {
            ray_count += 1;
            continue;
        }
        let coords = row[1..].iter().map(|v| *v / row[0]).collect::<Vec<_>>();
        vertices.push(coords);
    }

    stats.n_vertices = Some(vertices.len());
    stats.n_rays = Some(ray_count);
    stats.n_lines = Some(line_count);
    stats.bounded = Some(ray_count == 0 && line_count == 0);
    stats.affine_rank_vertices = Some(affine_rank(&vertices));
    stats
}

fn hrep_stats(matrix: &Matrix<f64, Inequality>) -> Stats {
    let mut stats = Stats::default();
    let eq_count = matrix.linearity().cardinality();
    stats.n_equalities = Some(eq_count);
    assert!(
        matrix.row_count() >= eq_count,
        "equality count exceeds total rows"
    );
    stats.n_inequalities = Some(matrix.row_count() - eq_count);
    stats
}

fn cdd_stats_from_matrix(matrix: &ParsedMatrix, label: &str) -> Result<Stats> {
    match matrix {
        ParsedMatrix::Inequality(m) => cdd_stats_from_inequality(m, label),
        ParsedMatrix::Generator(m) => cdd_stats_from_generator(m, label),
    }
}

fn cdd_stats_from_inequality(
    matrix: &Matrix<f64, hullabaloo::types::Inequality>,
    label: &str,
) -> Result<Stats> {
    let mut cdd: CddMatrix<f64, Inequality> =
        CddMatrix::new(matrix.row_count(), matrix.col_count(), NumberType::Real)
            .map_err(|e| anyhow!("cddlib matrix allocation failed for {label}: {e:?}"))?;
    for (r, row) in matrix.rows().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            cdd.set_real(r, c, val);
        }
    }
    let poly = CddPolyhedron::from_matrix(&cdd)
        .map_err(|e| anyhow!("cddlib conversion failed for {label}: {e:?}"))?;

    let facets = poly
        .facets()
        .map_err(|e| anyhow!("cddlib facets failed for {label}: {e:?}"))?;
    Ok(Stats {
        n_inequalities: Some(facets.rows()),
        n_equalities: Some(matrix.linearity().cardinality()),
        ..Default::default()
    })
}

fn cdd_stats_from_generator(matrix: &Matrix<f64, Generator>, label: &str) -> Result<Stats> {
    let mut cdd: CddMatrix<f64, Generator> =
        CddMatrix::new(matrix.row_count(), matrix.col_count(), NumberType::Real)
            .map_err(|e| anyhow!("cddlib matrix allocation failed for {label}: {e:?}"))?;
    for (r, row) in matrix.rows().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            cdd.set_real(r, c, val);
        }
        let is_vertex = !approx_eq(row[0], 0.0);
        cdd.set_generator_type(r, is_vertex);
    }
    let poly = CddPolyhedron::from_matrix(&cdd)
        .map_err(|e| anyhow!("cddlib conversion failed for {label}: {e:?}"))?;

    let gens = poly
        .generators()
        .map_err(|e| anyhow!("cddlib generators failed for {label}: {e:?}"))?;
    let mut stats = Stats::default();
    let mut vertices = Vec::new();
    let mut ray_count = 0usize;
    for r in 0..gens.rows() {
        let w = gens.get_real(r, 0);
        if approx_eq(w, 0.0) {
            ray_count += 1;
            continue;
        }
        let coords = (1..gens.cols())
            .map(|c| gens.get_real(r, c) / w)
            .collect::<Vec<_>>();
        vertices.push(coords);
    }
    stats.n_vertices = Some(vertices.len());
    stats.n_rays = Some(ray_count);
    stats.n_lines = Some(matrix.linearity().cardinality());
    stats.bounded = Some(ray_count == 0 && stats.n_lines == Some(0));
    stats.affine_rank_vertices = Some(affine_rank(&vertices));
    Ok(stats)
}

fn merge_stats(into: &mut Stats, extra: Stats) {
    if let Some(v) = extra.n_vertices {
        into.n_vertices.get_or_insert(v);
    }
    if let Some(v) = extra.n_inequalities {
        into.n_inequalities.get_or_insert(v);
    }
    if let Some(v) = extra.n_equalities {
        into.n_equalities.get_or_insert(v);
    }
    if let Some(v) = extra.n_rays {
        into.n_rays.get_or_insert(v);
    }
    if let Some(v) = extra.n_lines {
        into.n_lines.get_or_insert(v);
    }
    if let Some(v) = extra.bounded {
        into.bounded.get_or_insert(v);
    }
    if let Some(v) = extra.affine_rank_vertices {
        into.affine_rank_vertices.get_or_insert(v);
    }
}

fn assert_stats_match(expected: &Stats, actual: &Stats, item: &ItemSummary) {
    let label = item.id.as_deref().unwrap_or("<unknown polytope>");
    compare("n_vertices", expected.n_vertices, actual.n_vertices, label);
    compare(
        "n_inequalities",
        expected.n_inequalities,
        actual.n_inequalities,
        label,
    );
    compare(
        "n_equalities",
        expected.n_equalities,
        actual.n_equalities,
        label,
    );
    compare("n_rays", expected.n_rays, actual.n_rays, label);
    compare("n_lines", expected.n_lines, actual.n_lines, label);
    compare("bounded", expected.bounded, actual.bounded, label);
    compare(
        "affine_rank_vertices",
        expected.affine_rank_vertices,
        actual.affine_rank_vertices,
        label,
    );
}

fn compare<T: PartialEq + std::fmt::Debug>(
    name: &str,
    expected: Option<T>,
    actual: Option<T>,
    label: &str,
) {
    if let Some(exp) = expected {
        let act = actual.unwrap_or_else(|| panic!("missing {} for {}", name, label));
        assert_eq!(act, exp, "{} mismatch for {}", name, label);
    }
}
