use anyhow::{Result, anyhow, ensure};
use cddlib_rs::{Generator, Matrix as CddMatrix, NumberType, Polyhedron as CddPolyhedron};

use hullabaloo::AdjacencyList;

pub(super) fn build_cdd_polyhedron_rational(
    vertices: &[Vec<f64>],
) -> Result<CddPolyhedron<cddlib_rs::CddRational>> {
    ensure!(!vertices.is_empty(), "need at least one vertex");
    let dim = vertices[0].len();
    ensure!(dim > 0, "vertices must have positive dimension");
    ensure!(
        vertices.iter().skip(1).all(|v| v.len() == dim),
        "vertices must have consistent dimension"
    );

    let mut matrix: CddMatrix<cddlib_rs::CddRational, Generator> =
        CddMatrix::new(vertices.len(), dim + 1, NumberType::Rational)?;
    for (row, coords) in vertices.iter().enumerate() {
        matrix.set_generator_type(row, true);
        for (col, &v) in coords.iter().enumerate() {
            ensure!(v.is_finite(), "non-finite vertex coordinate {v}");
            matrix.set_real(row, col + 1, v);
        }
    }
    Ok(CddPolyhedron::from_matrix(&matrix)?)
}

pub(super) fn drum_width_from_vertices(
    vertices: &[Vec<f64>],
    top_count: usize,
    bot_count: usize,
) -> Result<usize> {
    ensure!(
        vertices.len() == top_count + bot_count,
        "drum vertex count mismatch: got={} top={} bot={}",
        vertices.len(),
        top_count,
        bot_count
    );

    let poly = build_cdd_polyhedron_rational(vertices)?;
    #[allow(deprecated)]
    let facet_graph =
        AdjacencyList::from_unsorted_adjacency_lists(poly.adjacency()?.to_adjacency_lists());
    let facets_to_vertices = poly.incidence()?.to_adjacency_lists();

    let mut is_top = vec![false; vertices.len()];
    let mut is_bot = vec![false; vertices.len()];
    is_top[..top_count].fill(true);
    is_bot[top_count..(top_count + bot_count)].fill(true);

    let mut top_facet = None;
    let mut bot_facet = None;
    for (facet_idx, verts) in facets_to_vertices.iter().enumerate() {
        if verts.len() == top_count
            && verts.iter().all(|&v| is_top[v])
            && top_facet.replace(facet_idx).is_some()
        {
            return Err(anyhow!("drum has multiple candidate top base facets"));
        }
        if verts.len() == bot_count
            && verts.iter().all(|&v| is_bot[v])
            && bot_facet.replace(facet_idx).is_some()
        {
            return Err(anyhow!("drum has multiple candidate bot base facets"));
        }
    }

    let top_facet = top_facet.ok_or_else(|| anyhow!("failed to locate drum top base facet"))?;
    let bot_facet = bot_facet.ok_or_else(|| anyhow!("failed to locate drum bot base facet"))?;

    facet_graph
        .distance(top_facet, bot_facet)
        .ok_or_else(|| anyhow!("graph distance not found"))
}

pub(super) fn drum_width<G: hullabaloo::Geometrizable<N = f64>>(
    geom: G,
    top_count: usize,
    bot_count: usize,
) -> Result<usize> {
    drum_width_from_vertices(&geom.into_vertices(), top_count, bot_count)
}
