use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use anyhow::{anyhow, ensure};
use calculo::num::Num;
use hullabaloo::adjacency::{AdjacencyBuilder, AdjacencyStore};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{DualRepresentation, Representation, RepresentationKind, RowSet};

use super::{
    AnyPolytopeCoefficients, CoefficientMatrix, CoefficientMode, RowMajorMatrix, Stats,
};

#[derive(Debug, Clone)]
pub(super) struct HowzatGeometrySummary<Inc, Adj> {
    pub(super) stats: Stats,
    pub(super) vertex_adjacency: Adj,
    pub(super) facets_to_vertices: Inc,
    pub(super) facet_adjacency: Adj,
    pub(super) facet_row_indices: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct HowzatExtractTimingDetail {
    pub(super) incidence: Duration,
    pub(super) vertex_adjacency: Duration,
    pub(super) facet_adjacency: Duration,
}

pub(super) fn summarize_howzat_geometry<
    Inc: From<SetFamily>,
    Adj: AdjacencyStore,
    N: Num,
    R: DualRepresentation,
>(
    howzat_poly: &howzat::polyhedron::PolyhedronOutput<N, R>,
    output_adjacency: bool,
    timing: bool,
    store_facet_row_indices: bool,
) -> Result<(HowzatGeometrySummary<Inc, Adj>, HowzatExtractTimingDetail), anyhow::Error> {
    let mut detail = HowzatExtractTimingDetail::default();
    let poly_dim = howzat_poly.dimension();
    ensure!(poly_dim > 0, "polyhedron dimension must be positive");

    match howzat_poly.representation() {
        RepresentationKind::Generator => {
            let start_incidence = timing.then(Instant::now);

            let generator_matrix = howzat_poly.input();
            let generator_rows = generator_matrix.row_count();
            let redundant_vertices = howzat_poly
                .redundant_rows()
                .cloned()
                .unwrap_or_else(|| RowSet::new(generator_rows));
            let dominant_vertices = howzat_poly
                .dominant_rows()
                .cloned()
                .unwrap_or_else(|| RowSet::new(generator_rows));

            let inequality_matrix = howzat_poly.output();
            let facet_rows = inequality_matrix.row_count();
            let facet_linearity = inequality_matrix.linearity().clone();
            let mut active_facets_mask = vec![true; facet_rows];
            for idx in facet_linearity.iter() {
                active_facets_mask[idx.as_index()] = false;
            }

            let incidence = howzat_poly
                .incidence()
                .ok_or_else(|| anyhow!("howzat incidence missing (output_incidence not requested)"))?;

            let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
            let mut old_to_new: Vec<Option<usize>> = vec![None; facet_rows];
            let active_facets = active_facets_mask.iter().filter(|&&a| a).count();
            let mut facets_to_vertices_rowsets: Vec<RowSet> = Vec::with_capacity(active_facets);
            let mut facet_row_indices =
                store_facet_row_indices.then(|| Vec::with_capacity(active_facets));

            for (old_idx, &active) in active_facets_mask.iter().enumerate() {
                if !active {
                    continue;
                }

                let face = incidence
                    .set(old_idx)
                    .cloned()
                    .unwrap_or_else(|| RowSet::new(generator_rows));
                let hash = face.hash_signature_u64();

                if let Some(candidates) = buckets.get(&hash)
                    && let Some(&existing_idx) = candidates
                        .iter()
                        .find(|&&idx| facets_to_vertices_rowsets[idx] == face)
                {
                    old_to_new[old_idx] = Some(existing_idx);
                    continue;
                }

                let new_idx = facets_to_vertices_rowsets.len();
                facets_to_vertices_rowsets.push(face);
                old_to_new[old_idx] = Some(new_idx);
                buckets.entry(hash).or_default().push(new_idx);
                if let Some(row_indices) = facet_row_indices.as_mut() {
                    row_indices.push(old_idx);
                }
            }

            if let Some(start) = start_incidence {
                detail.incidence = start.elapsed();
            }

            let vertex_count = generator_rows
                .saturating_sub(redundant_vertices.cardinality())
                .saturating_sub(dominant_vertices.cardinality());

            let vertex_adjacency = if output_adjacency {
                let start_vertex_adjacency = timing.then(Instant::now);
                let mut facets_by_vertex = vec![Vec::new(); generator_rows];
                for (facet_idx, face) in facets_to_vertices_rowsets.iter().enumerate() {
                    for v in face.iter().map(|id| id.as_index()) {
                        if redundant_vertices.contains(v) || dominant_vertices.contains(v) {
                            continue;
                        }
                        facets_by_vertex[v].push(facet_idx);
                    }
                }

                let vertex_adjacency =
                    hullabaloo::adjacency::input_adjacency_from_rows_by_node_with::<Adj::Builder>(
                        &facets_by_vertex,
                        facets_to_vertices_rowsets.len(),
                        &redundant_vertices,
                        &dominant_vertices,
                    );

                if let Some(start) = start_vertex_adjacency {
                    detail.vertex_adjacency = start.elapsed();
                }
                vertex_adjacency
            } else {
                Adj::Builder::new(0).finish()
            };

            let facet_count = facets_to_vertices_rowsets.len();
            let facet_adjacency = if output_adjacency {
                let start_facet_adjacency = timing.then(Instant::now);
                let facet_adjacency = if poly_dim >= 2 {
                    if let Some(facet_adjacency_sf) = howzat_poly.adjacency() {
                        let mut sink = Adj::Builder::new(facet_count);
                        for (old_i, neighbors) in facet_adjacency_sf.sets().iter().enumerate() {
                            let Some(new_i) = old_to_new.get(old_i).copied().flatten() else {
                                continue;
                            };
                            for old_j in neighbors.iter().map(|j| j.as_index()) {
                                let Some(new_j) = old_to_new.get(old_j).copied().flatten() else {
                                    continue;
                                };
                                if new_i < new_j {
                                    sink.add_undirected_edge(new_i, new_j);
                                }
                            }
                        }
                        sink.finish()
                    } else {
                        let mut facet_members: Vec<usize> = Vec::new();
                        let mut facet_offsets: Vec<usize> = Vec::with_capacity(facet_count + 1);
                        facet_offsets.push(0);
                        for face in &facets_to_vertices_rowsets {
                            facet_members.extend(face.iter().map(|id| id.as_index()));
                            facet_offsets.push(facet_members.len());
                        }
                        let rows_by_facet: Vec<&[usize]> = facet_offsets
                            .windows(2)
                            .map(|w| &facet_members[w[0]..w[1]])
                            .collect();
                        hullabaloo::adjacency::adjacency_from_rows_by_node_with::<Adj::Builder>(
                            &rows_by_facet,
                            generator_rows,
                            inequality_matrix.col_count(),
                            hullabaloo::adjacency::RowsByNodeAdjacencyOptions::default(),
                        )
                    }
                } else {
                    Adj::Builder::new(facet_count).finish()
                };

                if let Some(start) = start_facet_adjacency {
                    detail.facet_adjacency = start.elapsed();
                }
                facet_adjacency
            } else {
                Adj::Builder::new(0).finish()
            };

            let ridges = if output_adjacency {
                (0..facet_adjacency.node_count())
                    .map(|i| facet_adjacency.degree(i))
                    .sum::<usize>()
                    / 2
            } else {
                0
            };

            let facets_to_vertices: Inc =
                SetFamily::from_sets(generator_rows, facets_to_vertices_rowsets).into();

            Ok((
                HowzatGeometrySummary {
                    stats: Stats {
                        dimension: poly_dim,
                        vertices: vertex_count,
                        facets: facet_count,
                        ridges,
                    },
                    vertex_adjacency,
                    facets_to_vertices,
                    facet_adjacency,
                    facet_row_indices,
                },
                detail,
            ))
        }
        RepresentationKind::Inequality => {
            let start_incidence = timing.then(Instant::now);

            let inequality_matrix = howzat_poly.input();
            let facet_rows = inequality_matrix.row_count();
            let facet_linearity = inequality_matrix.linearity().clone();
            let redundant_facets = howzat_poly
                .redundant_rows()
                .cloned()
                .unwrap_or_else(|| RowSet::new(facet_rows));
            let dominant_facets = howzat_poly
                .dominant_rows()
                .cloned()
                .unwrap_or_else(|| RowSet::new(facet_rows));

            let mut old_to_active: Vec<Option<usize>> = vec![None; facet_rows];
            let mut active_facets = 0usize;
            let mut active_to_old = store_facet_row_indices.then(Vec::new);
            for old_idx in 0..facet_rows {
                if facet_linearity.contains(old_idx)
                    || redundant_facets.contains(old_idx)
                    || dominant_facets.contains(old_idx)
                {
                    continue;
                }
                old_to_active[old_idx] = Some(active_facets);
                active_facets += 1;
                if let Some(active_to_old) = active_to_old.as_mut() {
                    active_to_old.push(old_idx);
                }
            }

            let generator_matrix = howzat_poly.output();
            let vertex_rows = generator_matrix.row_count();

            let incidence = howzat_poly
                .incidence()
                .ok_or_else(|| anyhow!("howzat incidence missing (output_incidence not requested)"))?;

            let mut active_rowsets: Vec<RowSet> = (0..active_facets)
                .map(|_| RowSet::new(vertex_rows))
                .collect();
            for vertex_idx in 0..vertex_rows {
                let face = incidence
                    .set(vertex_idx)
                    .cloned()
                    .unwrap_or_else(|| RowSet::new(facet_rows));
                for facet in face.iter().map(|id| id.as_index()) {
                    let Some(active_idx) = old_to_active.get(facet).copied().flatten() else {
                        continue;
                    };
                    active_rowsets[active_idx].insert(vertex_idx);
                }
            }

            let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
            let mut active_to_new: Vec<Option<usize>> = vec![None; active_rowsets.len()];
            let mut facets_to_vertices_rowsets: Vec<RowSet> = Vec::new();
            facets_to_vertices_rowsets.reserve(active_rowsets.len());
            let mut facet_row_indices =
                store_facet_row_indices.then(|| Vec::with_capacity(active_rowsets.len()));

            for (active_idx, face) in active_rowsets.into_iter().enumerate() {
                let hash = face.hash_signature_u64();
                if let Some(candidates) = buckets.get(&hash)
                    && let Some(&existing_idx) = candidates
                        .iter()
                        .find(|&&idx| facets_to_vertices_rowsets[idx] == face)
                {
                    active_to_new[active_idx] = Some(existing_idx);
                    continue;
                }

                let new_idx = facets_to_vertices_rowsets.len();
                facets_to_vertices_rowsets.push(face);
                active_to_new[active_idx] = Some(new_idx);
                buckets.entry(hash).or_default().push(new_idx);
                if let Some(row_indices) = facet_row_indices.as_mut()
                    && let Some(active_to_old) = active_to_old.as_ref()
                {
                    row_indices.push(active_to_old[active_idx]);
                }
            }

            if let Some(start) = start_incidence {
                detail.incidence = start.elapsed();
            }

            drop(active_to_new);
            drop(old_to_active);

            let vertex_adjacency = if output_adjacency {
                let start_vertex_adjacency = timing.then(Instant::now);

                let facet_count = facets_to_vertices_rowsets.len();
                let mut facets_by_vertex = vec![Vec::new(); vertex_rows];
                for (facet_idx, face) in facets_to_vertices_rowsets.iter().enumerate() {
                    for v in face.iter().map(|id| id.as_index()) {
                        facets_by_vertex[v].push(facet_idx);
                    }
                }

                let vertex_adjacency =
                    hullabaloo::adjacency::input_adjacency_from_rows_by_node_with::<Adj::Builder>(
                        &facets_by_vertex,
                        facet_count,
                        &RowSet::new(vertex_rows),
                        &RowSet::new(vertex_rows),
                    );

                if let Some(start) = start_vertex_adjacency {
                    detail.vertex_adjacency = start.elapsed();
                }
                vertex_adjacency
            } else {
                Adj::Builder::new(0).finish()
            };

            let facet_count = facets_to_vertices_rowsets.len();
            let facet_adjacency = if output_adjacency {
                let start_facet_adjacency = timing.then(Instant::now);
                let facet_adjacency = if poly_dim >= 2 {
                    let mut facet_members: Vec<usize> = Vec::new();
                    let mut facet_offsets: Vec<usize> = Vec::with_capacity(facet_count + 1);
                    facet_offsets.push(0);
                    for face in &facets_to_vertices_rowsets {
                        facet_members.extend(face.iter().map(|id| id.as_index()));
                        facet_offsets.push(facet_members.len());
                    }
                    let rows_by_facet: Vec<&[usize]> = facet_offsets
                        .windows(2)
                        .map(|w| &facet_members[w[0]..w[1]])
                        .collect();
                    hullabaloo::adjacency::adjacency_from_rows_by_node_with::<Adj::Builder>(
                        &rows_by_facet,
                        vertex_rows,
                        inequality_matrix.col_count(),
                        hullabaloo::adjacency::RowsByNodeAdjacencyOptions::default(),
                    )
                } else {
                    Adj::Builder::new(facet_count).finish()
                };

                if let Some(start) = start_facet_adjacency {
                    detail.facet_adjacency = start.elapsed();
                }
                facet_adjacency
            } else {
                Adj::Builder::new(0).finish()
            };

            let ridges = if output_adjacency {
                (0..facet_adjacency.node_count())
                    .map(|i| facet_adjacency.degree(i))
                    .sum::<usize>()
                    / 2
            } else {
                0
            };

            let facet_count = facets_to_vertices_rowsets.len();
            let facets_to_vertices: Inc =
                SetFamily::from_sets(vertex_rows, facets_to_vertices_rowsets).into();
            let vertex_count = vertex_rows;

            Ok((
                HowzatGeometrySummary {
                    stats: Stats {
                        dimension: poly_dim,
                        vertices: vertex_count,
                        facets: facet_count,
                        ridges,
                    },
                    vertex_adjacency,
                    facets_to_vertices,
                    facet_adjacency,
                    facet_row_indices,
                },
                detail,
            ))
        }
    }
}

pub(super) fn extract_howzat_coefficients<N: super::CoefficientScalar, R: DualRepresentation>(
    howzat_poly: &howzat::polyhedron::PolyhedronOutput<N, R>,
    facet_row_indices: &[usize],
    coeff_mode: CoefficientMode,
) -> Result<Option<AnyPolytopeCoefficients>, anyhow::Error> {
    fn extract_coefficients_from_matrices<
        N: super::CoefficientScalar,
        G: Representation,
        H: Representation,
    >(
        generator_matrix: &howzat::matrix::LpMatrix<N, G>,
        inequality_matrix: &howzat::matrix::LpMatrix<N, H>,
        facet_row_indices: &[usize],
        coeff_mode: CoefficientMode,
    ) -> Result<Option<AnyPolytopeCoefficients>, anyhow::Error> {
        match coeff_mode {
            CoefficientMode::Off => Ok(None),
            CoefficientMode::F64 => {
                let gen_rows = generator_matrix.row_count();
                let gen_cols = generator_matrix.col_count();
                let mut gen_data = Vec::with_capacity(gen_rows.saturating_mul(gen_cols));
                for r in 0..gen_rows {
                    for v in generator_matrix.row(r).unwrap().iter() {
                        gen_data.push(v.to_f64());
                    }
                }

                let ineq_cols = inequality_matrix.col_count();
                let mut ineq_data = Vec::with_capacity(
                    facet_row_indices.len().saturating_mul(ineq_cols),
                );
                for &row in facet_row_indices {
                    for v in inequality_matrix.row(row).unwrap().iter() {
                        ineq_data.push(v.to_f64());
                    }
                }

                Ok(Some(AnyPolytopeCoefficients {
                    generators: CoefficientMatrix::F64(RowMajorMatrix {
                        rows: gen_rows,
                        cols: gen_cols,
                        data: gen_data,
                    }),
                    inequalities: CoefficientMatrix::F64(RowMajorMatrix {
                        rows: facet_row_indices.len(),
                        cols: ineq_cols,
                        data: ineq_data,
                    }),
                }))
            }
            CoefficientMode::Exact => {
                let gen_rows = generator_matrix.row_count();
                let gen_cols = generator_matrix.col_count();
                let mut gen_data: Vec<N> = Vec::with_capacity(gen_rows.saturating_mul(gen_cols));
                for r in 0..gen_rows {
                    for v in generator_matrix.row(r).unwrap().iter() {
                        gen_data.push(v.clone());
                    }
                }

                let ineq_cols = inequality_matrix.col_count();
                let mut ineq_data: Vec<N> = Vec::with_capacity(
                    facet_row_indices.len().saturating_mul(ineq_cols),
                );
                for &row in facet_row_indices {
                    for v in inequality_matrix.row(row).unwrap().iter() {
                        ineq_data.push(v.clone());
                    }
                }

                Ok(Some(AnyPolytopeCoefficients {
                    generators: CoefficientMatrix::from_num(gen_rows, gen_cols, gen_data),
                    inequalities: CoefficientMatrix::from_num(
                        facet_row_indices.len(),
                        ineq_cols,
                        ineq_data,
                    ),
                }))
            }
        }
    }

    match howzat_poly.representation() {
        RepresentationKind::Generator => extract_coefficients_from_matrices(
            howzat_poly.input(),
            howzat_poly.output(),
            facet_row_indices,
            coeff_mode,
        ),
        RepresentationKind::Inequality => extract_coefficients_from_matrices(
            howzat_poly.output(),
            howzat_poly.input(),
            facet_row_indices,
            coeff_mode,
        ),
    }
}
