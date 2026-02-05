use std::collections::BTreeMap;
use std::sync::Mutex;

use cddlib_rs as cdd;
use hullabaloo::adjacency::{IncidenceAdjacencyOptions, adjacency_from_incidence};
use hullabaloo::types::{Generator, RepresentationKind};
use lrslib_rs as lrs;

const EPS: f64 = 1e-9;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().expect("test serial lock poisoned")
}

fn eval_inequality(row: &[f64], x: &[f64]) -> f64 {
    assert_eq!(row.len(), x.len() + 1);
    let mut v = row[0];
    for j in 0..x.len() {
        v += row[j + 1] * x[j];
    }
    v
}

fn lrs_row(m: &lrs::Matrix, r: usize) -> Vec<f64> {
    (0..m.cols()).map(|c| m.get(r, c)).collect()
}

fn cdd_row(m: &cdd::Matrix, r: usize) -> Vec<f64> {
    (0..m.cols()).map(|c| m.get_real(r, c)).collect()
}

fn facet_tight_vertices(row: &[f64], vertices: &[Vec<f64>]) -> Vec<usize> {
    let dim = row.len() - 1;
    let mut tight = Vec::new();
    for (i, v) in vertices.iter().enumerate() {
        assert_eq!(v.len(), dim);
        let val = eval_inequality(row, v);
        assert!(
            val >= -EPS,
            "vertex {i} violates inequality: value={val} row={row:?} v={v:?}"
        );
        if val.abs() <= EPS {
            tight.push(i);
        }
    }
    tight
}

fn facets_as_tight_sets_lrs(vertices: &[Vec<f64>], facets: &lrs::Matrix) -> Vec<Vec<usize>> {
    assert_eq!(facets.representation(), RepresentationKind::Inequality);
    let mut sets: Vec<Vec<usize>> = (0..facets.rows())
        .map(|r| facet_tight_vertices(&lrs_row(facets, r), vertices))
        .collect();
    for s in &mut sets {
        s.sort_unstable();
        s.dedup();
    }
    sets.sort();
    sets
}

fn facets_as_tight_sets_cdd(vertices: &[Vec<f64>], facets: &cdd::Matrix) -> Vec<Vec<usize>> {
    assert_eq!(facets.representation(), RepresentationKind::Inequality);
    let mut sets: Vec<Vec<usize>> = (0..facets.rows())
        .map(|r| facet_tight_vertices(&cdd_row(facets, r), vertices))
        .collect();
    for s in &mut sets {
        s.sort_unstable();
        s.dedup();
    }
    sets.sort();
    sets
}

fn generator_vertices_lrs(g: &lrs::Matrix) -> Vec<Vec<f64>> {
    assert_eq!(g.representation(), RepresentationKind::Generator);
    let dim = g.cols() - 1;
    let mut out = Vec::new();
    for r in 0..g.rows() {
        if g.get(r, 0) != 1.0 {
            continue;
        }
        out.push((0..dim).map(|j| g.get(r, j + 1)).collect());
    }
    out
}

fn generator_vertices_cdd<N: cdd::CddNumber>(g: &cdd::Matrix<N, Generator>) -> Vec<Vec<f64>> {
    let dim = g.cols() - 1;
    let mut out = Vec::new();
    for r in 0..g.rows() {
        if g.get_real(r, 0) != 1.0 {
            continue;
        }
        out.push((0..dim).map(|j| g.get_real(r, j + 1)).collect());
    }
    out
}

fn vertex_tight_inequalities(h_rows: &[Vec<f64>], x: &[f64]) -> Vec<usize> {
    let mut tight = Vec::new();
    for (i, row) in h_rows.iter().enumerate() {
        let val = eval_inequality(row, x);
        assert!(
            val >= -EPS,
            "vertex violates input inequality {i}: value={val} row={row:?} v={x:?}"
        );
        if val.abs() <= EPS {
            tight.push(i);
        }
    }
    tight
}

fn vertices_as_tight_sets(vertices: &[Vec<f64>], h_rows: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let mut sets: Vec<Vec<usize>> = vertices
        .iter()
        .map(|v| {
            let mut s = vertex_tight_inequalities(h_rows, v);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    sets.sort();
    sets
}

fn adjacency_by_ids(
    ids: &[Vec<usize>],
    adj: &[Vec<usize>],
) -> BTreeMap<Vec<usize>, Vec<Vec<usize>>> {
    assert_eq!(ids.len(), adj.len());
    let mut out = BTreeMap::new();
    for (i, key) in ids.iter().enumerate() {
        let mut neigh: Vec<Vec<usize>> = adj[i].iter().map(|&j| ids[j].clone()).collect();
        neigh.sort();
        neigh.dedup();
        out.insert(key.clone(), neigh);
    }
    out
}

fn unique_ids(ids: &[Vec<usize>]) {
    let mut sorted = ids.to_vec();
    sorted.sort();
    let mut prev: Option<&[usize]> = None;
    for id in &sorted {
        if let Some(p) = prev {
            assert_ne!(p, id.as_slice(), "duplicate id {id:?}");
        }
        prev = Some(id);
    }
}

#[test]
fn square_v_to_h_facets_match_cddlib() {
    let _serial = serial();

    let verts = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];

    let lrs_poly = lrs::Polyhedron::from_vertices(&verts).unwrap();
    let lrs_facets = lrs_poly.facets().unwrap();

    let cdd_mat = cdd::Matrix::from_vertex_rows(&verts).unwrap();
    let cdd_poly = cdd::Polyhedron::from_matrix(&cdd_mat).unwrap();
    let cdd_facets = cdd_poly.facets().unwrap();

    assert_eq!(lrs_facets.cols(), cdd_facets.cols());
    assert_eq!(lrs_facets.rows(), cdd_facets.rows());

    let lrs_sets = facets_as_tight_sets_lrs(&verts, &lrs_facets);
    let cdd_sets = facets_as_tight_sets_cdd(&verts, &cdd_facets);
    assert_eq!(lrs_sets, cdd_sets);
}

#[test]
fn square_h_to_v_vertices_match_cddlib() {
    let _serial = serial();

    // Unit square: x>=0, y>=0, x<=1, y<=1.
    let h_rows = vec![
        vec![0.0, 1.0, 0.0],  // x >= 0
        vec![0.0, 0.0, 1.0],  // y >= 0
        vec![1.0, -1.0, 0.0], // 1 - x >= 0
        vec![1.0, 0.0, -1.0], // 1 - y >= 0
    ];

    let lrs_h = lrs::Matrix::from_rows(&h_rows, RepresentationKind::Inequality).unwrap();
    let lrs_poly = lrs::Polyhedron::from_matrix(lrs_h);
    let lrs_gens = lrs_poly.generators().unwrap();
    let lrs_vertices = generator_vertices_lrs(&lrs_gens);

    let mut cdd_h: cdd::Matrix =
        cdd::Matrix::new(h_rows.len(), h_rows[0].len(), cdd::NumberType::Rational).unwrap();
    for (r, row) in h_rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            cdd_h.set_real(r, c, v);
        }
    }
    let cdd_poly = cdd::Polyhedron::from_matrix(&cdd_h).unwrap();
    let cdd_gens = cdd_poly.generators().unwrap();
    let cdd_vertices = generator_vertices_cdd(&cdd_gens);

    assert_eq!(lrs_vertices.len(), cdd_vertices.len());
    let lrs_ids = vertices_as_tight_sets(&lrs_vertices, &h_rows);
    let cdd_ids = vertices_as_tight_sets(&cdd_vertices, &h_rows);
    assert_eq!(lrs_ids, cdd_ids);
}

#[test]
fn square_vertex_adjacency_matches_cddlib() {
    let _serial = serial();

    let verts = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];

    let lrs_poly = lrs::Polyhedron::from_vertices(&verts).unwrap();
    let lrs_solved = lrs_poly.solve().unwrap();
    let lrs_adj = adjacency_from_incidence(
        lrs_solved.input_incidence().sets(),
        lrs_solved.output().rows(),
        lrs_solved.input().cols(),
        IncidenceAdjacencyOptions::default(),
    )
    .into_adjacency_lists();

    let cdd_mat = cdd::Matrix::from_vertex_rows(&verts).unwrap();
    let cdd_poly = cdd::Polyhedron::from_matrix(&cdd_mat).unwrap();
    let cdd_adj = cdd_poly.input_adjacency().unwrap().to_adjacency_lists();

    assert_eq!(lrs_adj, cdd_adj);
}

#[test]
fn square_facet_adjacency_matches_cddlib() {
    let _serial = serial();

    let verts = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];

    let lrs_poly = lrs::Polyhedron::from_vertices(&verts).unwrap();
    let lrs_solved = lrs_poly.solve().unwrap();
    let lrs_facets = lrs_solved.output();
    let lrs_ids: Vec<Vec<usize>> = (0..lrs_facets.rows())
        .map(|r| {
            let mut s = facet_tight_vertices(&lrs_row(lrs_facets, r), &verts);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    unique_ids(&lrs_ids);
    let lrs_adj = adjacency_from_incidence(
        lrs_solved.incidence().sets(),
        lrs_solved.incidence().universe_size(),
        lrs_solved.input().cols(),
        IncidenceAdjacencyOptions::default(),
    )
    .into_adjacency_lists();
    let lrs_adj_canon = adjacency_by_ids(&lrs_ids, &lrs_adj);

    let cdd_mat = cdd::Matrix::from_vertex_rows(&verts).unwrap();
    let cdd_poly = cdd::Polyhedron::from_matrix(&cdd_mat).unwrap();
    let cdd_facets = cdd_poly.facets().unwrap();
    let cdd_ids: Vec<Vec<usize>> = (0..cdd_facets.rows())
        .map(|r| {
            let mut s = facet_tight_vertices(&cdd_row(&cdd_facets, r), &verts);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    unique_ids(&cdd_ids);
    #[allow(deprecated)]
    let cdd_adj = cdd_poly.adjacency().unwrap().to_adjacency_lists();
    let cdd_adj_canon = adjacency_by_ids(&cdd_ids, &cdd_adj);

    assert_eq!(lrs_adj_canon, cdd_adj_canon);
}

#[test]
fn tetrahedron_parity_with_cddlib() {
    let _serial = serial();

    let verts = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    // V -> H facets parity.
    let lrs_poly = lrs::Polyhedron::from_vertices(&verts).unwrap();
    let lrs_facets = lrs_poly.facets().unwrap();
    let cdd_mat = cdd::Matrix::from_vertex_rows(&verts).unwrap();
    let cdd_poly = cdd::Polyhedron::from_matrix(&cdd_mat).unwrap();
    let cdd_facets = cdd_poly.facets().unwrap();

    assert_eq!(lrs_facets.rows(), cdd_facets.rows());
    assert_eq!(lrs_facets.cols(), cdd_facets.cols());

    let lrs_sets = facets_as_tight_sets_lrs(&verts, &lrs_facets);
    let cdd_sets = facets_as_tight_sets_cdd(&verts, &cdd_facets);
    assert_eq!(lrs_sets, cdd_sets);

    // H -> V vertices parity + vertex adjacency parity (canonicalized by tight inequalities).
    let h_rows = vec![
        vec![0.0, 1.0, 0.0, 0.0],    // x >= 0
        vec![0.0, 0.0, 1.0, 0.0],    // y >= 0
        vec![0.0, 0.0, 0.0, 1.0],    // z >= 0
        vec![1.0, -1.0, -1.0, -1.0], // 1 - x - y - z >= 0
    ];

    let lrs_h = lrs::Matrix::from_rows(&h_rows, RepresentationKind::Inequality).unwrap();
    let lrs_cols = lrs_h.cols();
    let lrs_poly_h = lrs::Polyhedron::from_matrix(lrs_h);
    let lrs_solved_h = lrs_poly_h.solve().unwrap();
    let lrs_gens = lrs_solved_h.output();
    let lrs_vertices = generator_vertices_lrs(lrs_gens);
    let lrs_vids: Vec<Vec<usize>> = lrs_vertices
        .iter()
        .map(|v| {
            let mut s = vertex_tight_inequalities(&h_rows, v);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    unique_ids(&lrs_vids);
    let lrs_adj = adjacency_from_incidence(
        lrs_solved_h.incidence().sets(),
        lrs_solved_h.incidence().universe_size(),
        lrs_cols,
        IncidenceAdjacencyOptions::default(),
    )
    .into_adjacency_lists();
    let lrs_adj_canon = adjacency_by_ids(&lrs_vids, &lrs_adj);

    let mut cdd_h: cdd::Matrix =
        cdd::Matrix::new(h_rows.len(), h_rows[0].len(), cdd::NumberType::Rational).unwrap();
    for (r, row) in h_rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            cdd_h.set_real(r, c, v);
        }
    }
    let cdd_poly_h = cdd::Polyhedron::from_matrix(&cdd_h).unwrap();
    let cdd_gens = cdd_poly_h.generators().unwrap();
    let cdd_vertices = generator_vertices_cdd(&cdd_gens);
    let cdd_vids: Vec<Vec<usize>> = cdd_vertices
        .iter()
        .map(|v| {
            let mut s = vertex_tight_inequalities(&h_rows, v);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    unique_ids(&cdd_vids);
    #[allow(deprecated)]
    let cdd_adj = cdd_poly_h.adjacency().unwrap().to_adjacency_lists();
    let cdd_adj_canon = adjacency_by_ids(&cdd_vids, &cdd_adj);

    // Vertex sets must match.
    let mut lrs_sorted = lrs_vids.clone();
    lrs_sorted.sort();
    let mut cdd_sorted = cdd_vids.clone();
    cdd_sorted.sort();
    assert_eq!(lrs_sorted, cdd_sorted);

    // And adjacency must match up to canonical vertex IDs.
    assert_eq!(lrs_adj_canon, cdd_adj_canon);
}

#[test]
fn cube_parity_with_cddlib() {
    let _serial = serial();

    // Unit cube [0,1]^3.
    let verts = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];

    // V -> H facets parity + facet adjacency parity (canonicalized by tight vertices).
    let lrs_poly = lrs::Polyhedron::from_vertices(&verts).unwrap();
    let lrs_solved = lrs_poly.solve().unwrap();
    let lrs_facets = lrs_solved.output();
    let lrs_ids: Vec<Vec<usize>> = (0..lrs_facets.rows())
        .map(|r| {
            let mut s = facet_tight_vertices(&lrs_row(lrs_facets, r), &verts);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    unique_ids(&lrs_ids);
    let lrs_adj = adjacency_from_incidence(
        lrs_solved.incidence().sets(),
        lrs_solved.incidence().universe_size(),
        lrs_solved.input().cols(),
        IncidenceAdjacencyOptions::default(),
    )
    .into_adjacency_lists();
    let lrs_adj_canon = adjacency_by_ids(&lrs_ids, &lrs_adj);

    let cdd_mat = cdd::Matrix::from_vertex_rows(&verts).unwrap();
    let cdd_poly = cdd::Polyhedron::from_matrix(&cdd_mat).unwrap();
    let cdd_facets = cdd_poly.facets().unwrap();
    let cdd_ids: Vec<Vec<usize>> = (0..cdd_facets.rows())
        .map(|r| {
            let mut s = facet_tight_vertices(&cdd_row(&cdd_facets, r), &verts);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    unique_ids(&cdd_ids);
    #[allow(deprecated)]
    let cdd_adj = cdd_poly.adjacency().unwrap().to_adjacency_lists();
    let cdd_adj_canon = adjacency_by_ids(&cdd_ids, &cdd_adj);

    assert_eq!(lrs_ids.len(), 6);
    assert_eq!(cdd_ids.len(), 6);
    assert_eq!(lrs_adj_canon, cdd_adj_canon);

    // H -> V vertices parity + vertex adjacency parity (canonicalized by tight inequalities).
    let h_rows = vec![
        vec![0.0, 1.0, 0.0, 0.0],  // x >= 0
        vec![0.0, 0.0, 1.0, 0.0],  // y >= 0
        vec![0.0, 0.0, 0.0, 1.0],  // z >= 0
        vec![1.0, -1.0, 0.0, 0.0], // 1 - x >= 0
        vec![1.0, 0.0, -1.0, 0.0], // 1 - y >= 0
        vec![1.0, 0.0, 0.0, -1.0], // 1 - z >= 0
    ];

    let lrs_h = lrs::Matrix::from_rows(&h_rows, RepresentationKind::Inequality).unwrap();
    let lrs_cols = lrs_h.cols();
    let lrs_poly_h = lrs::Polyhedron::from_matrix(lrs_h);
    let lrs_solved_h = lrs_poly_h.solve().unwrap();
    let lrs_gens = lrs_solved_h.output();
    let lrs_vertices = generator_vertices_lrs(lrs_gens);
    let lrs_vids: Vec<Vec<usize>> = lrs_vertices
        .iter()
        .map(|v| {
            let mut s = vertex_tight_inequalities(&h_rows, v);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    unique_ids(&lrs_vids);
    let lrs_vadj = adjacency_from_incidence(
        lrs_solved_h.incidence().sets(),
        lrs_solved_h.incidence().universe_size(),
        lrs_cols,
        IncidenceAdjacencyOptions::default(),
    )
    .into_adjacency_lists();
    let lrs_vadj_canon = adjacency_by_ids(&lrs_vids, &lrs_vadj);

    let mut cdd_h: cdd::Matrix =
        cdd::Matrix::new(h_rows.len(), h_rows[0].len(), cdd::NumberType::Rational).unwrap();
    for (r, row) in h_rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            cdd_h.set_real(r, c, v);
        }
    }
    let cdd_poly_h = cdd::Polyhedron::from_matrix(&cdd_h).unwrap();
    let cdd_gens = cdd_poly_h.generators().unwrap();
    let cdd_vertices = generator_vertices_cdd(&cdd_gens);
    let cdd_vids: Vec<Vec<usize>> = cdd_vertices
        .iter()
        .map(|v| {
            let mut s = vertex_tight_inequalities(&h_rows, v);
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();
    unique_ids(&cdd_vids);
    #[allow(deprecated)]
    let cdd_vadj = cdd_poly_h.adjacency().unwrap().to_adjacency_lists();
    let cdd_vadj_canon = adjacency_by_ids(&cdd_vids, &cdd_vadj);

    assert_eq!(lrs_vids.len(), 8);
    assert_eq!(cdd_vids.len(), 8);
    assert_eq!(lrs_vadj_canon, cdd_vadj_canon);
}
