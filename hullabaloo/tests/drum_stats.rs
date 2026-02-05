use std::collections::BTreeMap;

use anyhow::{Result, anyhow, ensure};
use cddlib_rs::{Generator, Polyhedron};
use hullabaloo::{Drum, Geometrizable};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

const DRUM_SEEDS: &[u64] = &[1, 2, 3, 5, 8];
const DRUM_TOP_DIM: usize = 7;
const DRUM_TOP_VERTICES: usize = 12;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Stats {
    dimension: usize,
    vertices: usize,
    facets: usize,
    ridges: usize,
}

#[derive(Clone, Debug)]
struct RandomDrumConfig {
    top_dim: usize,
    top_vertices: usize,
    seed: u64,
}

impl Default for RandomDrumConfig {
    fn default() -> Self {
        Self {
            top_dim: 3,
            top_vertices: 8,
            seed: 0,
        }
    }
}

impl RandomDrumConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.top_dim > 0, "random-drum: n must be > 0");
        ensure!(
            self.top_vertices > self.top_dim,
            "random-drum: v must be at least n+1 for full dimension"
        );
        ensure!(
            self.top_dim > 1 || self.top_vertices <= 2,
            "random-drum: one-dimensional tops support exactly 2 vertices"
        );
        Ok(())
    }
}

fn init_rng(seed: u64) -> (u64, StdRng) {
    if seed == 0 {
        let seed = rand::rng().random();
        (seed, StdRng::seed_from_u64(seed))
    } else {
        (seed, StdRng::seed_from_u64(seed))
    }
}

fn sample_random_vertices(
    dim: usize,
    count: usize,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, usize)> {
    if dim == 1 && count > 2 {
        return Err(anyhow!(
            "cannot sample {count} non-redundant vertices in 1D (maximum is 2)"
        ));
    }

    const SCALE: f64 = 10.0;

    let mut verts: Vec<Vec<f64>> = Vec::with_capacity(count);
    let mut matrix: Option<cddlib_rs::Matrix<f64, Generator>> = None;
    let mut attempts: usize = 0;

    while verts.len() < count {
        attempts += 1;
        let candidate: Vec<f64> = (0..dim).map(|_| rng.random_range(-SCALE..SCALE)).collect();

        let new_row = cddlib_rs::Matrix::<f64, Generator>::from_vertex_rows(std::slice::from_ref(
            &candidate,
        ))?;

        if let Some(existing) = &mut matrix {
            existing.append_rows_in_place(&new_row)?;
            if existing.redundant_rows()?.is_empty() {
                verts.push(candidate);
            } else {
                existing.remove_row(existing.rows() - 1)?;
            }
        } else if new_row.redundant_rows()?.is_empty() {
            matrix = Some(new_row);
            verts.push(candidate);
        }
    }

    Ok((verts, attempts))
}

fn standard_simplex_vertices(dim: usize) -> Vec<Vec<f64>> {
    let mut verts = Vec::with_capacity(dim + 1);
    verts.push(vec![0.0; dim]);
    for i in 0..dim {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        verts.push(v);
    }
    verts
}

fn drum_sample(top_dim: usize, top_vertices: usize, seed: u64) -> Result<Vec<Vec<f64>>> {
    let cfg = RandomDrumConfig {
        top_dim,
        top_vertices,
        seed,
    };
    cfg.validate()?;
    let (_seed, mut rng) = init_rng(cfg.seed);
    let (top_vertices, _) = sample_random_vertices(cfg.top_dim, cfg.top_vertices, &mut rng)?;
    let bot_vertices = standard_simplex_vertices(cfg.top_dim);
    let drum = Drum::<f64>::new(top_vertices, bot_vertices);
    Ok(drum.into_vertices())
}

fn stats_from_poly(poly: &Polyhedron<f64>) -> Result<Stats> {
    let facets = poly.facets()?;
    let dim = facets.cols().saturating_sub(1);
    #[allow(deprecated)]
    let facet_graph = poly.adjacency()?.to_adjacency_lists();
    let ridges = facet_graph.iter().map(|n| n.len()).sum::<usize>() / 2;
    let vertices = poly.generators()?.rows();
    Ok(Stats {
        dimension: dim,
        vertices,
        facets: facets.rows(),
        ridges,
    })
}

fn expected_stats_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("drum_expected.json")
}

fn load_expected() -> Result<BTreeMap<u64, Stats>> {
    let path = expected_stats_path();
    let contents = std::fs::read_to_string(path)?;
    let map = serde_json::from_str(&contents)?;
    Ok(map)
}

#[test]
fn drums_match_reference_stats() -> Result<()> {
    let expected = load_expected()?;
    for &seed in DRUM_SEEDS {
        let vertices = drum_sample(DRUM_TOP_DIM, DRUM_TOP_VERTICES, seed)?;
        let poly = Polyhedron::<f64>::from_vertex_rows(&vertices)?;
        let actual = stats_from_poly(&poly)?;
        let reference = expected
            .get(&seed)
            .ok_or_else(|| anyhow!("missing expected stats for seed {seed}"))?;
        assert_eq!(
            actual, *reference,
            "stats mismatch for seed {seed}: actual {:?} expected {:?}",
            actual, reference
        );
    }
    Ok(())
}
