//! Safe, idiomatic Rust bindings on top of the raw `ppl-sys` FFI.
//!
//! The primary goal is V→H conversion (convex hull facets) plus facet↔vertex incidence.

mod raw;

use calculo::num::f64_row_to_scaled_ints;
use rug::Integer;
use thiserror::Error;

pub use hullabaloo::set_family::ListFamily;

#[derive(Debug, Error)]
pub enum PplError {
    #[error("invalid matrix dimensions (rows={rows}, cols={cols})")]
    InvalidMatrix { rows: usize, cols: usize },

    #[error("matrix rows are ragged (inconsistent column counts)")]
    RaggedMatrix,

    #[error("non-finite value {value} at row={row} col={col}")]
    NonFinite { row: usize, col: usize, value: f64 },

    #[error("ppl error: {0}")]
    Ppl(&'static str),

    #[error("ppl call failed: {0}")]
    PplCallFailed(String),
}

pub type PplResult<T> = Result<T, PplError>;

#[derive(Debug, Clone)]
pub struct Inequalities {
    rows: usize,
    cols: usize,
    data: Vec<Integer>,
}

impl Inequalities {
    #[inline(always)]
    pub fn row_count(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    pub fn col_count(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    pub fn row(&self, row: usize) -> &[Integer] {
        assert!(row < self.rows);
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline(always)]
    pub fn rows(&self) -> std::iter::Take<std::slice::ChunksExact<'_, Integer>> {
        self.data.chunks_exact(self.cols).take(self.rows)
    }

    pub fn into_flat(self) -> Vec<Integer> {
        self.data
    }
}

#[derive(Debug, Clone)]
pub struct Polyhedron {
    vertices_exact: Vec<VertexExact>,
    dim: usize,
}

impl Polyhedron {
    pub fn from_vertices(vertices: Vec<Vec<f64>>) -> PplResult<Self> {
        let Some(first) = vertices.first() else {
            return Err(PplError::InvalidMatrix { rows: 0, cols: 0 });
        };
        let dim = first.len();
        if dim == 0 {
            return Err(PplError::InvalidMatrix {
                rows: vertices.len(),
                cols: 0,
            });
        }
        if vertices.iter().skip(1).any(|v| v.len() != dim) {
            return Err(PplError::RaggedMatrix);
        }
        for (row, vtx) in vertices.iter().enumerate() {
            for (col, &value) in vtx.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PplError::NonFinite { row, col, value });
                }
            }
        }

        let vertices_exact = precompute_vertices_exact(&vertices, dim);
        Ok(Self { vertices_exact, dim })
    }

    pub fn from_integer_vertices(vertices: Vec<Vec<i64>>) -> PplResult<Self> {
        let Some(first) = vertices.first() else {
            return Err(PplError::InvalidMatrix { rows: 0, cols: 0 });
        };
        let dim = first.len();
        if dim == 0 {
            return Err(PplError::InvalidMatrix {
                rows: vertices.len(),
                cols: 0,
            });
        }
        if vertices.iter().skip(1).any(|v| v.len() != dim) {
            return Err(PplError::RaggedMatrix);
        }

        let mut vertices_exact = Vec::with_capacity(vertices.len());
        for vtx in vertices {
            let mut coords_scaled = Vec::with_capacity(dim);
            for value in vtx {
                coords_scaled.push(Integer::from(value));
            }
            vertices_exact.push(VertexExact {
                denom: Integer::from(1),
                coords_scaled,
            });
        }

        Ok(Self { vertices_exact, dim })
    }

    pub fn from_integer_rows(vertices: &[&[i64]]) -> PplResult<Self> {
        let Some(first) = vertices.first() else {
            return Err(PplError::InvalidMatrix { rows: 0, cols: 0 });
        };
        let dim = first.len();
        if dim == 0 {
            return Err(PplError::InvalidMatrix {
                rows: vertices.len(),
                cols: 0,
            });
        }
        if vertices.iter().skip(1).any(|v| v.len() != dim) {
            return Err(PplError::RaggedMatrix);
        }

        let mut vertices_exact = Vec::with_capacity(vertices.len());
        for vtx in vertices {
            let mut coords_scaled = Vec::with_capacity(dim);
            for &value in *vtx {
                coords_scaled.push(Integer::from(value));
            }
            vertices_exact.push(VertexExact {
                denom: Integer::from(1),
                coords_scaled,
            });
        }

        Ok(Self { vertices_exact, dim })
    }

    pub fn solve(&self) -> PplResult<PolyhedronSolved> {
        raw::ensure_initialized().map_err(PplError::PplCallFailed)?;

        let (facet_to_vertex, facet_coeffs) =
            raw::convex_hull_facets_incidence(&self.vertices_exact, self.dim)?;
        let facet_count = facet_to_vertex.len();
        let incidence = ListFamily::from_sorted_sets(facet_to_vertex, self.vertices_exact.len());
        let input_incidence = incidence.transpose();

        Ok(PolyhedronSolved {
            dim: self.dim,
            vertices: self.vertices_exact.len(),
            facets: facet_count,
            inequalities: Inequalities {
                rows: facet_count,
                cols: self.dim.saturating_add(1),
                data: facet_coeffs,
            },
            incidence,
            input_incidence,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PolyhedronSolved {
    dim: usize,
    vertices: usize,
    facets: usize,
    inequalities: Inequalities,
    incidence: ListFamily,
    input_incidence: ListFamily,
}

impl PolyhedronSolved {
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices
    }

    pub fn facet_count(&self) -> usize {
        self.facets
    }

    pub fn inequalities(&self) -> &Inequalities {
        &self.inequalities
    }

    pub fn incidence(&self) -> &ListFamily {
        &self.incidence
    }

    pub fn input_incidence(&self) -> &ListFamily {
        &self.input_incidence
    }

    pub fn into_parts(self) -> (usize, usize, usize, Inequalities, ListFamily, ListFamily) {
        (
            self.dim,
            self.vertices,
            self.facets,
            self.inequalities,
            self.incidence,
            self.input_incidence,
        )
    }
}

#[derive(Debug, Clone)]
struct VertexExact {
    denom: Integer,
    coords_scaled: Vec<Integer>,
}

fn precompute_vertices_exact(vertices: &[Vec<f64>], dim: usize) -> Vec<VertexExact> {
    let mut out = Vec::with_capacity(vertices.len());
    for vtx in vertices {
        debug_assert_eq!(vtx.len(), dim, "vertex dimension mismatch");
        let (denom, coords_scaled) = f64_row_to_scaled_ints::<Integer>(vtx);
        out.push(VertexExact {
            denom,
            coords_scaled,
        });
    }
    out
}
